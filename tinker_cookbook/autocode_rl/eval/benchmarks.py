from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Literal, cast

from datasets import Dataset, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.autocode_rl.eval.common import BenchmarkAdapter, EvalExample, GradeResult
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.sandbox import SandboxBackend

logger = logging.getLogger(__name__)

GPQAConfigName = Literal["gpqa_diamond", "gpqa_main", "gpqa_extended", "gpqa_experts"]


def _limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return cast(Dataset, dataset.select(range(limit)))


def _lookup_required_text(row: dict[str, Any], candidates: list[str]) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            value = lowered[candidate.lower()]
            if value is not None:
                return str(value)
    raise KeyError(f"Unable to find any of {candidates} in row keys: {list(row.keys())}")


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def _collect_choice_candidates(response_text: str) -> list[str]:
    candidates: list[str] = []

    for match in re.findall(r"\\boxed\{\s*([A-Da-d])\s*\}", response_text):
        candidates.append(match.upper())

    for match in re.findall(
        r"(?im)(?:final answer|answer)\s*(?:is|:)?\s*\(?([A-Da-d])\)?(?:[.)]|$)",
        response_text,
    ):
        candidates.append(match.upper())

    non_empty_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if non_empty_lines:
        last_line = non_empty_lines[-1]
        line_match = re.fullmatch(r"\(?([A-Da-d])\)?[.)]?", last_line)
        if line_match is not None:
            candidates.append(line_match.group(1).upper())

    return _unique_keep_order(candidates)


def _collect_math_candidates(
    response_text: str,
    *,
    prefer_integer: bool,
    extract_boxed_fn: Any,
) -> list[str]:
    candidates: list[str] = []

    try:
        boxed = str(extract_boxed_fn(response_text)).strip()
        if boxed:
            candidates.append(boxed)
    except Exception:
        pass

    for match in re.findall(
        r"(?im)(?:final answer|answer)\s*(?:is|:)?\s*(.+?)(?:$|\n)",
        response_text,
    ):
        candidate = match.strip().rstrip(".")
        if candidate:
            candidates.append(candidate)

    non_empty_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if non_empty_lines:
        candidates.append(non_empty_lines[-1].rstrip("."))
        if len(non_empty_lines) >= 2:
            candidates.append(non_empty_lines[-2].rstrip("."))

    if prefer_integer:
        for match in re.findall(r"(?<![A-Za-z0-9])-?\d+(?![A-Za-z0-9])", response_text):
            candidates.append(match)

    return _unique_keep_order([candidate for candidate in candidates if candidate])


def _get_math_grading_helpers() -> tuple[Any, Any]:
    try:
        from tinker_cookbook.recipes.math_rl.math_grading import (
            extract_boxed,
            grade_answer,
            grade_answer_math_verify,
            run_with_timeout_signal,
        )
    except ImportError as exc:
        raise ConfigurationError(
            "AIME/Math500 eval requires the math-rl extras. "
            "Install them with: uv pip install -e '.[math-rl]'"
        ) from exc

    def safe_grade(
        given_answer: str,
        ground_truth: str,
        grader: Literal["sympy", "math_verify"],
        timeout: float,
    ) -> bool:
        if grader == "sympy":
            grader_func = grade_answer
        elif grader == "math_verify":
            grader_func = grade_answer_math_verify
        else:
            raise ValueError(f"Invalid grader: {grader}")

        result = run_with_timeout_signal(
            grader_func,
            args=(given_answer, ground_truth),
            timeout_seconds=max(1, int(timeout)),
        )
        return bool(result)

    return extract_boxed, safe_grade


def _standard_math_fewshot_prefix() -> list[renderers.Message]:
    return [
        {
            "role": "user",
            "content": "How many r's are in strawberry? Write your answer in \\boxed{} format.",
        },
        {
            "role": "assistant",
            "content": (
                "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w "
                "6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}"
            ),
        },
    ]


@dataclass(frozen=True)
class MathBenchmarkAdapter(BenchmarkAdapter):
    name: str
    dataset_name: str
    split: str
    answer_suffix: str
    integer_answer: bool = False
    use_standard_fewshot: bool = False
    math_grader: Literal["sympy", "math_verify"] = "sympy"
    math_grader_timeout: float = 1.0

    def load_examples(self, limit: int | None = None) -> list[EvalExample]:
        _get_math_grading_helpers()
        dataset = cast(Dataset, load_dataset(self.dataset_name, split=self.split))
        dataset = _limit_dataset(dataset, limit)

        examples: list[EvalExample] = []
        for index, row in enumerate(dataset):
            example = cast(dict[str, Any], row)
            problem = _lookup_required_text(example, ["problem", "question"])
            answer = self._extract_ground_truth(example)
            messages = self._build_messages(problem)
            example_id = str(example.get("id", f"{self.name}_{index:04d}"))
            examples.append(
                EvalExample(
                    index=index,
                    example_id=example_id,
                    messages=messages,
                    payload=answer,
                    metadata={
                        "dataset_name": self.dataset_name,
                        "split": self.split,
                        "ground_truth": answer,
                    },
                )
            )

        return examples

    async def grade_response(self, example: EvalExample, response_text: str) -> GradeResult:
        extract_boxed, safe_grade = _get_math_grading_helpers()

        ground_truth = cast(str, example.payload)
        candidates = _collect_math_candidates(
            response_text,
            prefer_integer=self.integer_answer,
            extract_boxed_fn=extract_boxed,
        )
        if not candidates:
            return GradeResult(correct=False, extracted=False, prediction=None, details={})

        for candidate in candidates:
            if safe_grade(candidate, ground_truth, self.math_grader, self.math_grader_timeout):
                return GradeResult(
                    correct=True,
                    extracted=True,
                    prediction=candidate,
                    details={"candidates": candidates[:5]},
                )

        return GradeResult(
            correct=False,
            extracted=True,
            prediction=candidates[0],
            details={"candidates": candidates[:5]},
        )

    def describe(self) -> dict[str, str | bool | float]:
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "integer_answer": self.integer_answer,
            "use_standard_fewshot": self.use_standard_fewshot,
            "math_grader": self.math_grader,
            "math_grader_timeout": self.math_grader_timeout,
        }

    def _build_messages(self, problem: str) -> list[renderers.Message]:
        messages: list[renderers.Message] = []
        if self.use_standard_fewshot:
            messages.extend(_standard_math_fewshot_prefix())
        messages.append({"role": "user", "content": f"{problem}{self.answer_suffix}"})
        return messages

    def _extract_ground_truth(self, row: dict[str, Any]) -> str:
        answer = row.get("answer")
        if answer is not None and str(answer).strip():
            return str(answer).strip()

        solution = row.get("solution")
        if solution is None:
            raise ValueError(f"Unable to find answer/solution fields in row keys: {list(row.keys())}")

        extract_boxed, _safe_grade = _get_math_grading_helpers()
        return str(extract_boxed(str(solution))).strip()


@dataclass(frozen=True)
class AIMEBenchmarkAdapter(MathBenchmarkAdapter):
    def __init__(
        self,
        *,
        split: str,
        use_standard_fewshot: bool,
        math_grader: Literal["sympy", "math_verify"],
        math_grader_timeout: float,
    ):
        super().__init__(
            name="aime",
            dataset_name="HuggingFaceH4/aime_2024",
            split=split,
            answer_suffix="\n\nThe final answer is an integer from 0 to 999. Write your final answer in \\boxed{} format.",
            integer_answer=True,
            use_standard_fewshot=use_standard_fewshot,
            math_grader=math_grader,
            math_grader_timeout=math_grader_timeout,
        )

    def _extract_ground_truth(self, row: dict[str, Any]) -> str:
        answer = row.get("answer")
        if answer is not None and str(answer).strip():
            return str(answer).strip()
        return super()._extract_ground_truth(row)


@dataclass(frozen=True)
class Math500BenchmarkAdapter(MathBenchmarkAdapter):
    def __init__(
        self,
        *,
        split: str,
        use_standard_fewshot: bool,
        math_grader: Literal["sympy", "math_verify"],
        math_grader_timeout: float,
    ):
        super().__init__(
            name="math500",
            dataset_name="HuggingFaceH4/MATH-500",
            split=split,
            answer_suffix=" Write your answer in \\boxed{} format.",
            integer_answer=False,
            use_standard_fewshot=use_standard_fewshot,
            math_grader=math_grader,
            math_grader_timeout=math_grader_timeout,
        )


@dataclass(frozen=True)
class GPQABenchmarkAdapter(BenchmarkAdapter):
    name: str = "gpqa"
    split: str = "train"
    gpqa_config: GPQAConfigName = "gpqa_diamond"
    shuffle_choices: bool = True
    seed: int = 0

    def load_examples(self, limit: int | None = None) -> list[EvalExample]:
        dataset = cast(
            Dataset,
            load_dataset("Idavidrein/gpqa", name=self.gpqa_config, split=self.split),
        )
        dataset = _limit_dataset(dataset, limit)

        examples: list[EvalExample] = []
        for index, row in enumerate(dataset):
            example = cast(dict[str, Any], row)
            question = _lookup_required_text(example, ["Question", "question"])
            correct_answer = _lookup_required_text(
                example,
                ["Correct Answer", "correct answer", "correct_answer"],
            )
            incorrect_answers = self._extract_incorrect_answers(example)
            option_pairs = [(correct_answer, True)] + [(answer, False) for answer in incorrect_answers]
            if self.shuffle_choices:
                random.Random(f"{self.seed}:{index}").shuffle(option_pairs)

            letters = "ABCD"
            options = {
                letters[i]: option_pairs[i][0]
                for i in range(len(option_pairs))
            }
            correct_letter = next(
                letter for letter, (_answer, is_correct) in zip(letters, option_pairs, strict=True) if is_correct
            )
            example_id = str(example.get("Record ID", example.get("record_id", f"gpqa_{index:04d}")))
            domain = str(example.get("High-level domain", example.get("high_level_domain", "")))
            subdomain = str(example.get("Subdomain", example.get("subdomain", "")))
            prompt = self._build_prompt(question, options)

            examples.append(
                EvalExample(
                    index=index,
                    example_id=example_id,
                    messages=[{"role": "user", "content": prompt}],
                    payload=correct_letter,
                    metadata={
                        "dataset_name": "Idavidrein/gpqa",
                        "split": self.split,
                        "gpqa_config": self.gpqa_config,
                        "correct_letter": correct_letter,
                        "domain": domain,
                        "subdomain": subdomain,
                        "options": dict(options),
                    },
                )
            )

        return examples

    async def grade_response(self, example: EvalExample, response_text: str) -> GradeResult:
        correct_letter = cast(str, example.payload)
        candidates = _collect_choice_candidates(response_text)
        if not candidates:
            return GradeResult(correct=False, extracted=False, prediction=None, details={})

        return GradeResult(
            correct=correct_letter in candidates,
            extracted=True,
            prediction=candidates[0],
            details={"candidates": candidates[:5], "correct_letter": correct_letter},
        )

    def describe(self) -> dict[str, str | bool | int]:
        return {
            "dataset_name": "Idavidrein/gpqa",
            "split": self.split,
            "gpqa_config": self.gpqa_config,
            "shuffle_choices": self.shuffle_choices,
            "seed": self.seed,
        }

    def _extract_incorrect_answers(self, row: dict[str, Any]) -> list[str]:
        incorrect_answers: list[str] = []
        for key in sorted(row):
            if key.lower().startswith("incorrect answer"):
                value = row[key]
                if value is not None and str(value).strip():
                    incorrect_answers.append(str(value).strip())
        if len(incorrect_answers) != 3:
            raise ValueError(
                f"Expected exactly 3 incorrect answers, found {len(incorrect_answers)} for row keys {list(row.keys())}"
            )
        return incorrect_answers

    def _build_prompt(self, question: str, options: dict[str, str]) -> str:
        option_lines = "\n".join(f"{letter}. {text}" for letter, text in options.items())
        return (
            "Answer the following GPQA multiple-choice question.\n"
            "Choose exactly one option. You may reason, but end with a single boxed letter: "
            "\\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}.\n\n"
            f"Question:\n{question}\n\n"
            f"{option_lines}"
        )


@dataclass(frozen=True)
class LCBBenchmarkAdapter(BenchmarkAdapter):
    name: str = "lcb"
    split: Literal["train", "test"] = "test"
    sandbox_backend: SandboxBackend | None = None
    sandbox_timeout: int = 6
    seed: int = 0

    def load_examples(self, limit: int | None = None) -> list[EvalExample]:
        from tinker_cookbook.recipes.autocode_rl.autocode_env import LCBTask, load_lcb_tasks

        tasks = load_lcb_tasks(split=self.split, seed=self.seed)
        if limit is not None:
            tasks = tasks[:limit]

        examples: list[EvalExample] = []
        for index, task in enumerate(tasks):
            lcb_task = cast(LCBTask, task)
            examples.append(
                EvalExample(
                    index=index,
                    example_id=f"lcb_{index:04d}",
                    messages=[{"role": "user", "content": lcb_task.problem}],
                    payload=lcb_task,
                    metadata={
                        "split": self.split,
                        "num_tests": len(lcb_task.tests),
                        "has_starter_code": bool(lcb_task.starter_code),
                    },
                )
            )

        return examples

    async def grade_response(self, example: EvalExample, response_text: str) -> GradeResult:
        from tinker_cookbook.recipes.autocode_rl.autocode_env import LCBTask
        from tinker_cookbook.recipes.code_rl.code_grading import (
            extract_code_from_model,
            sandbox_check_correctness,
        )

        task = cast(LCBTask, example.payload)
        code = extract_code_from_model(response_text)
        if code is None:
            return GradeResult(correct=False, extracted=False, prediction=None, details={})

        passed, details = await sandbox_check_correctness(
            task.tests,
            code,
            timeout=self.sandbox_timeout,
            backend=self.sandbox_backend,
        )
        return GradeResult(
            correct=bool(passed),
            extracted=True,
            prediction=code,
            details={"sandbox_details": cast(Any, details)},
        )

    def describe(self) -> dict[str, str | int | None]:
        return {
            "split": self.split,
            "sandbox_backend": None if self.sandbox_backend is None else str(self.sandbox_backend),
            "sandbox_timeout": self.sandbox_timeout,
            "seed": self.seed,
        }
