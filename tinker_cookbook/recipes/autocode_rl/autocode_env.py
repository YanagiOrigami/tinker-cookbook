"""
AutoCode RL — single-turn competitive-programming environment.

Training & eval-judge use a remote judge server (see ``autocode_grading``).
An optional LCB eval mode runs LiveCodeBench problems in a local sandbox
(same pipeline as ``code_rl``).

Prompt style
~~~~~~~~~~~~
The model receives an LCB-style prompt that explicitly allows **Python or C++**.
See :func:`build_autocode_prompt` for the full template.

Data format (JSONL, one object per line)::

    {"problem_id": "1234A", "problem_statement": "Given an integer n, ..."}

Evaluation modes (``AutocodeDatasetBuilder.eval_mode``):

- ``"judge"`` — loads ``test_path`` JSONL, grades via remote judge (default).
- ``"lcb"``   — loads DeepCoder/LiveCodeBench from HuggingFace, grades via
  local sandbox execution (SandboxFusion or Modal).
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import chz
import tinker
from datasets import Dataset, concatenate_datasets, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.autocode_rl.autocode_grading import (
    extract_code_from_response,
    judge_code,
)
from tinker_cookbook.recipes.autocode_rl.lcb_utils import fetch_live_code_bench_system_prompt
from tinker_cookbook.recipes.code_rl.code_grading import (
    extract_code_from_model,
    sandbox_check_correctness,
)
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

logger = logging.getLogger(__name__)


# ===================================================================
# Shared task types
# ===================================================================


@dataclass(frozen=True)
class AutocodeTask:
    """A competitive-programming task submitted to a remote judge.

    Attributes:
        problem_id: Unique identifier used by the judge to look up test cases.
        problem_statement: Raw problem text (will be wrapped by
            :func:`build_autocode_prompt` before being shown to the model).
    """

    problem_id: str
    problem_statement: str


@dataclass(frozen=True)
class LCBTask:
    """A LiveCodeBench task graded by local sandbox execution (Python only).

    Attributes:
        problem: Full prompt including the LCB system message.
        tests: Normalized test cases for ``sandbox_check_correctness``.
        starter_code: Optional starter code snippet.
    """

    problem: str
    tests: list[dict[str, Any]]
    starter_code: str | None = None


# ===================================================================
# Question formatting (LCB-style, but allows Python or C++)
# ===================================================================

_AUTOCODE_SYSTEM_MESSAGE = (
    "You are an expert programmer. You will be given a question "
    "(problem specification) and will generate a correct program "
    "that matches the specification and passes all tests. "
    "You should write your solution in C++."
)

_AUTOCODE_FORMATTING = (
    "Read the inputs from stdin, solve the problem, and write the answer to stdout "
    "(do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows."
)


def build_autocode_prompt(problem_statement: str) -> str:
    """Build the full prompt for a competitive-programming task.

    Mirrors the LiveCodeBench prompt structure but accepts only C++.
    """
    prompt = _AUTOCODE_SYSTEM_MESSAGE + "\n\n"
    prompt += problem_statement + "\n\n"
    prompt += f"### Format: {_AUTOCODE_FORMATTING}\n"
    prompt += "```cpp\n// YOUR CODE HERE\n```\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


_FEWSHOT_PREFIX: list[renderers.Message] = [
    {
        "role": "user",
        "content": build_autocode_prompt(
            'Write a program that reads 2 integers a and b from stdin and prints their sum.'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "```cpp\n"
            "#include <iostream>\n"
            "using namespace std;\n\n"
            "int main() {\n"
            "    int a, b;\n"
            "    cin >> a >> b;\n"
            "    cout << a + b << endl;\n"
            "    return 0;\n"
            "}\n"
            "```"
        ),
    },
]


def _build_question(task: AutocodeTask) -> str:
    return build_autocode_prompt(task.problem_statement)


# ===================================================================
# AutocodeEnv — remote judge
# ===================================================================


class AutocodeEnv(Env):
    """Single-turn environment graded by a remote judge.

    The model receives an LCB-style prompt (via :func:`build_autocode_prompt`)
    that allows Python or C++.  The response is parsed for a fenced code block,
    which is submitted to the remote judge configured by ``JUDGE_HOST`` /
    ``JUDGE_PORT`` environment variables.

    Reward formula::

        reward = format_coef * (correct_format - 1) + correct
    """

    def __init__(
        self,
        task: AutocodeTask,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
    ):
        self.task = task
        self.renderer = renderer
        self.convo_prefix = convo_prefix or []
        self.format_coef = format_coef

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[renderers.base.Observation, StopCondition]:
        convo = self.convo_prefix + [
            {"role": "user", "content": _build_question(self.task)},
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        convo = self.convo_prefix + [
            {"role": "user", "content": _build_question(self.task)},
        ]
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        # --- format check ---
        extracted = extract_code_from_response(content)
        has_code_block = extracted is not None
        correct_format = float(parse_success) and float(has_code_block)

        # --- correctness via remote judge ---
        if extracted is not None:
            code, lang = extracted
            try:
                passed, _details = await judge_code(code, lang, self.task.problem_id)
                correct = float(passed)
            except Exception as e:
                logger.warning("Judge error for %s: %s", self.task.problem_id, e)
                correct = 0.0
        else:
            correct = 0.0

        reward = self.format_coef * (correct_format - 1.0) + correct

        # --- logging ---
        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=convo))
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "problem_id": self.task.problem_id,
                    "format_valid": bool(correct_format),
                    "correct": bool(correct),
                    "format_coef": self.format_coef,
                    "reward": f"{reward:.3f}",
                },
                caption="Reward components",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct,
            },
        )


# ===================================================================
# LCBEvalEnv — local sandbox (LiveCodeBench)
# ===================================================================


class LCBEvalEnv(Env):
    """Single-turn environment graded by local sandbox execution.

    Uses the same ``sandbox_check_correctness`` pipeline as ``code_rl``.
    Problems come from the LiveCodeBench / DeepCoder HuggingFace dataset
    and are presented with the standard LCB Python-only prompt.  Code is
    executed in a SandboxFusion (Docker) or Modal sandbox.
    """

    def __init__(
        self,
        task: LCBTask,
        renderer: renderers.Renderer,
        sandbox_backend: SandboxBackend | None = None,
        timeout: int = 6,
        format_coef: float = 0.1,
    ):
        self.task = task
        self.renderer = renderer
        self.sandbox_backend = sandbox_backend
        self.timeout = timeout
        self.format_coef = format_coef

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[renderers.base.Observation, StopCondition]:
        convo: list[renderers.Message] = [
            {"role": "user", "content": self.task.problem},
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        code = extract_code_from_model(content)
        has_code_block = code is not None
        correct_format = float(parse_success) and float(has_code_block)

        if code is not None:
            try:
                passed, _details = await sandbox_check_correctness(
                    self.task.tests,
                    code,
                    timeout=self.timeout,
                    backend=self.sandbox_backend,
                )
                correct = float(passed)
            except Exception as e:
                logger.warning("Sandbox error: %s", e)
                correct = 0.0
        else:
            correct = 0.0

        reward = self.format_coef * (correct_format - 1.0) + correct

        with logtree.scope_header("Prompt"):
            logtree.log_text(self.task.problem[:500])
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "format_valid": bool(correct_format),
                    "correct": bool(correct),
                    "format_coef": self.format_coef,
                    "reward": f"{reward:.3f}",
                },
                caption="Reward components",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct,
            },
        )


# ===================================================================
# EnvGroupBuilders
# ===================================================================


@dataclass(frozen=True)
class AutocodeGroupBuilder(EnvGroupBuilder):
    """Creates a group of :class:`AutocodeEnv` instances for one task.

    All envs in the group share the same task; the training loop samples
    ``num_envs`` independent rollouts and uses group-level variance for
    advantage estimation.
    """

    task: AutocodeTask
    renderer: renderers.Renderer
    num_envs: int
    convo_prefix: list[renderers.Message] | None = None
    format_coef: float = 0.1

    async def make_envs(self) -> Sequence[Env]:
        return [
            AutocodeEnv(
                task=self.task,
                renderer=self.renderer,
                convo_prefix=self.convo_prefix,
                format_coef=self.format_coef,
            )
            for _ in range(self.num_envs)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["autocode"]


@dataclass(frozen=True)
class LCBEvalGroupBuilder(EnvGroupBuilder):
    """Creates a group of :class:`LCBEvalEnv` instances for one LCB task.

    Typically used with ``num_envs=1`` for evaluation.
    """

    task: LCBTask
    renderer: renderers.Renderer
    num_envs: int
    sandbox_backend: SandboxBackend | None = None
    timeout: int = 6
    format_coef: float = 0.1

    async def make_envs(self) -> Sequence[Env]:
        return [
            LCBEvalEnv(
                task=self.task,
                renderer=self.renderer,
                sandbox_backend=self.sandbox_backend,
                timeout=self.timeout,
                format_coef=self.format_coef,
            )
            for _ in range(self.num_envs)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["lcb_eval"]


# ===================================================================
# Data loading
# ===================================================================


def load_tasks_from_jsonl(path: str, seed: int | None = None) -> list[AutocodeTask]:
    """Load tasks from a local JSONL file.

    Each line must have ``problem_id`` and ``problem_statement`` fields.
    """
    ds = load_dataset("json", data_files=path, split="train")
    ds = cast(Dataset, ds)
    if seed is not None:
        ds = ds.shuffle(seed=seed)

    tasks: list[AutocodeTask] = []
    for row in ds:
        row = cast(dict[str, Any], row)
        pid = row.get("problem_id")
        stmt = row.get("problem_statement")
        if pid is None or stmt is None:
            logger.warning("Skipping row missing problem_id or problem_statement")
            continue
        tasks.append(AutocodeTask(problem_id=str(pid), problem_statement=str(stmt)))

    logger.info("Loaded %d tasks from %s", len(tasks), path)
    return tasks


def _ensure_dict(val: Any) -> dict[str, Any]:
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return {}
    return val if isinstance(val, dict) else {}


def _normalize_lcb_tests(raw_tests: Any, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize test cases to the format expected by sandbox_check_correctness."""
    tests = raw_tests
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError:
            return []
    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        # TACO-style → per-case list
        from tinker_cookbook.recipes.code_rl.code_grading import taco_to_lcb_format

        tests = taco_to_lcb_format(tests)
    if isinstance(tests, dict):
        tests = [tests]

    normalized: list[dict[str, Any]] = []
    for test in tests or []:
        if not isinstance(test, dict):
            continue
        testtype = test.get("testtype") or "stdin_stdout"
        test_metadata = _ensure_dict(test.get("metadata", {}))
        if testtype == "functional":
            func_name = test_metadata.get("func_name") or metadata.get("func_name")
            if func_name is not None:
                test_metadata["func_name"] = str(func_name)
        normalized.append(
            {
                "input": str(test.get("input", "")),
                "output": str(test.get("output", "")),
                "testtype": testtype,
                "metadata": test_metadata or {"func_name": None},
            }
        )
    return normalized


def load_lcb_tasks(
    split: Literal["train", "test"] = "test",
    seed: int = 0,
) -> list[LCBTask]:
    """Load tasks from the DeepCoder HF dataset for LCB-style sandbox eval.

    Uses the same dataset / normalization logic as ``code_rl``.
    """
    # if split == "train":
    #     names = ("primeintellect", "taco", "lcbv5")
    # else:
    #     names = ("codeforces", "lcbv5")
    names = ("lcbv5",)

    datasets = []
    for name in names:
        logger.info("Loading DeepCoder/%s split=%s ...", name, split)
        ds = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split=split)
        datasets.append(cast(Dataset, ds))
    ds = cast(Dataset, concatenate_datasets(datasets))

    if split == "train":
        ds = ds.shuffle(seed=seed)

    tasks: list[LCBTask] = []
    for item in ds:
        row = cast(dict[str, Any], item)
        metadata = _ensure_dict(row.get("metadata", {}))

        raw_tests = row.get("tests") or row.get("ground_truth")
        tests = _normalize_lcb_tests(raw_tests, metadata)
        if not tests:
            continue

        question = row.get("question") or row.get("prompt") or row.get("problem")
        if not isinstance(question, str) or not question.strip():
            continue

        starter_code = row.get("starter_code")
        if isinstance(starter_code, str) and starter_code.strip():
            problem = fetch_live_code_bench_system_prompt(question, starter_code)
        else:
            problem = fetch_live_code_bench_system_prompt(question)
            starter_code = None

        tasks.append(
            LCBTask(
                problem=problem,
                tests=tests,
                starter_code=starter_code if isinstance(starter_code, str) else None,
            )
        )

    logger.info("Loaded %d LCB tasks (split=%s)", len(tasks), split)
    return tasks


# ===================================================================
# RLDataset (generic for both builder types)
# ===================================================================


class AutocodeDataset(RLDataset):
    """Iterates over EnvGroupBuilders in batches.

    Args:
        builders: List of environment group builders.
        batch_size: Number of builders per batch.
        recurrent: How many times to repeat the dataset.  For eval this
            should be ``1`` so evaluation terminates after one pass.  For
            training it can be set higher (default ``1``); use
            ``max_steps`` on the training config to control duration
            instead of inflating the dataset length.
        seed: Base seed for the per-epoch permutation.  Each epoch uses
            ``random.Random(seed + epoch)`` to shuffle builder indices,
            so order varies across epochs but stays reproducible.
    """

    def __init__(
        self,
        builders: list[EnvGroupBuilder],
        batch_size: int,
        recurrent: int = 1,
        seed: int = 0,
    ):
        self.builders = builders
        self.batch_size = batch_size
        self.recurrent = recurrent
        self._seed = seed
        self._perms: dict[int, list[int]] = {}

    def _epoch_perm(self, epoch: int) -> list[int]:
        if epoch not in self._perms:
            rng = random.Random(self._seed + epoch)
            perm = list(range(len(self.builders)))
            rng.shuffle(perm)
            self._perms[epoch] = perm
        return self._perms[epoch]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        num_batches = self.num_unique_batches
        epoch = index // num_batches
        wrapped = index % num_batches
        perm = self._epoch_perm(epoch)
        start = wrapped * self.batch_size
        end = min(start + self.batch_size, len(self.builders))
        return [self.builders[i] for i in perm[start:end]]

    @property
    def num_unique_batches(self) -> int:
        return math.ceil(len(self.builders) / self.batch_size)

    def __len__(self) -> int:
        return self.num_unique_batches * self.recurrent


# ===================================================================
# RLDatasetBuilder
# ===================================================================


@chz.chz
class AutocodeDatasetBuilder(RLDatasetBuilder):
    """Build train & eval datasets for AutoCode RL.

    **Train split** always loads from a local JSONL file (``train_path``) and
    grades via the remote judge.

    **Eval split** is controlled by ``eval_mode``:

    - ``"judge"`` (default) — loads ``test_path`` JSONL, grades via the same
      remote judge as training.
    - ``"lcb"`` — loads DeepCoder / LiveCodeBench problems from HuggingFace
      and grades Python solutions in a local sandbox (SandboxFusion or Modal).
      Requires a running sandbox; see README for setup.
    """

    model_name_for_tokenizer: str
    renderer_name: str
    batch_size: int
    group_size: int
    train_path: str = "data/train.jsonl"
    test_path: str = "data/eval.jsonl"
    eval_mode: Literal["judge", "lcb"] = "judge"
    sandbox_backend: SandboxBackend | None = None
    sandbox_timeout: int = 6
    format_coef: float = 0.1
    convo_prefix: Literal["standard"] | None = "standard"
    seed: int = 0
    train_recurrent: int = 1

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        prefix = _FEWSHOT_PREFIX if self.convo_prefix == "standard" else None

        # --- train (always remote-judge based) ---
        train_tasks = load_tasks_from_jsonl(self.train_path, seed=self.seed)
        train_builders: list[EnvGroupBuilder] = [
            AutocodeGroupBuilder(
                task=t,
                renderer=renderer,
                num_envs=self.group_size,
                convo_prefix=prefix,
                format_coef=self.format_coef,
            )
            for t in train_tasks
        ]
        train_dataset = AutocodeDataset(
            train_builders,
            self.batch_size,
            recurrent=self.train_recurrent,
            seed=self.seed,
        )

        # --- test ---
        if self.eval_mode == "lcb":
            test_dataset = self._build_lcb_eval(renderer)
        else:
            test_dataset = self._build_judge_eval(renderer, prefix)

        return train_dataset, test_dataset

    def _build_judge_eval(
        self,
        renderer: renderers.Renderer,
        prefix: list[renderers.Message] | None,
    ) -> AutocodeDataset:
        test_tasks = load_tasks_from_jsonl(self.test_path)
        test_builders: list[EnvGroupBuilder] = [
            AutocodeGroupBuilder(
                task=t,
                renderer=renderer,
                num_envs=1,
                convo_prefix=prefix,
                format_coef=self.format_coef,
            )
            for t in test_tasks
        ]
        return AutocodeDataset(test_builders, self.batch_size)

    def _build_lcb_eval(self, renderer: renderers.Renderer) -> AutocodeDataset:
        lcb_tasks = load_lcb_tasks("test", seed=self.seed)
        test_builders: list[EnvGroupBuilder] = [
            LCBEvalGroupBuilder(
                task=t,
                renderer=renderer,
                num_envs=1,
                sandbox_backend=self.sandbox_backend,
                timeout=self.sandbox_timeout,
                format_coef=self.format_coef,
            )
            for t in lcb_tasks
        ]
        return AutocodeDataset(test_builders, self.batch_size)
