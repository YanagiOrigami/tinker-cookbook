import math
import re
from functools import partial
from typing import Literal, Sequence, cast

import chz
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.coding_rl.coding_grading import (
    remote_code_judge,
    extract_code,
)
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

class CodingEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        id: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        timeout: float = 60.0,
    ):
        super().__init__(renderer, convo_prefix)
        self.problem = problem
        self.id = id
        self.timeout = timeout

    @classmethod
    def question_suffix(cls) -> str:
        return "Please write the correct code to solve the above problem, remember to put code inside a code box like ```<lang>\n<your code>\n```. Do not output anything else after the code box. Your solution should be executable and correct."

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        try:
            answer, lang = extract_code(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        try:
            answer, lang = extract_code(sample_str)
        except ValueError:
            return False
        return remote_code_judge(sample_str, self.id)
    
    def get_reference_answer(self) -> str:
        return self.id

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": "Write a program that prints \"Hello, World!\"." + CodingEnv.question_suffix(),
            },
            {
                "role": "assistant",
                "content": "```cpp\n#include <iostream>\n\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}\n```",
            },
        ]

def _get_lcbp_coding_test() -> Dataset:
    test_dataset = load_dataset("QAQAQAQAQ/LiveCodeBench-Pro", split="quater_2025_7_9")
    return cast(Dataset, test_dataset)

def _get_lcbp_coding_train() -> Dataset:
    test_problems: set[str] = {
        problem["problem_statement"]  # pyright: ignore[reportArgumentType, reportCallIssue]
        for problem in _get_lcbp_coding_test()
    }

    dataset_name = "QAQAQAQAQ/LiveCodeBench-Pro"
    ds1 = load_dataset(dataset_name, split="biannual_2024_7_12")
    ds2 = load_dataset(dataset_name, split="biannual_2025_1_6")
    full_dataset = concatenate_datasets([ds1, ds2])
    return full_dataset

def _load_from_local_jsonl(path: str, split: str) -> Dataset:
    files_to_load = {split: path}
    dataset = load_dataset("json", data_files=files_to_load)
    return cast(Dataset, dataset[split])

class CodingDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        epochs: int = 1,
    ):
        if split == "train":
            self.ds = _get_lcbp_coding_train().shuffle(seed=seed)
        elif split == "test":
            self.ds = _get_lcbp_coding_test()
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.epochs = epochs

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        index = index % (( len(self.ds) + self.batch_size - 1) // self.batch_size)
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]
    
    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size) * self.epochs

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        id = x["problem_id"]
        return ProblemGroupBuilder(
            env_thunk=partial(
                CodingEnv, x["problem_statement"], id, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
        )
    
class LocalCodingDataset(CodingDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        epochs: int = 1,
    ):
        if split == "train":
            self.ds = _load_from_local_jsonl("/mnt/e/Research/RL_everything/data/output/train.jsonl", "train").shuffle(seed=seed)
        elif split == "test":
            self.ds = _load_from_local_jsonl("/mnt/e/Research/RL_everything/data/output/eval.jsonl", "test")
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.epochs = epochs

@chz.chz
class CodingDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    epochs: int = 1

    async def __call__(self) -> tuple[CodingDataset, CodingDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = CodingEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            CodingDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
                epochs=self.epochs if split == "train" else 1,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])
    
@chz.chz    
class LocalCodingDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    epochs: int = 1

    async def __call__(self) -> tuple[LocalCodingDataset, LocalCodingDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = CodingEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            LocalCodingDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
                epochs=self.epochs if split == "train" else 1,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


DATASET_BUILDER_MAP = {
    "coding": CodingDatasetBuilder,
    "coding_local": LocalCodingDatasetBuilder,
}

def get_coding_dataset_builder(
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
    epochs: int = 1,
) -> RLDatasetBuilder:
    if dataset_name not in DATASET_BUILDER_MAP:
        raise ValueError(f"Unknown coding dataset: {dataset_name}. Available: {list(DATASET_BUILDER_MAP.keys())}")

    builder_class = DATASET_BUILDER_MAP[dataset_name]

    return builder_class(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        group_size=group_size,
        seed=seed,
        epochs=epochs,
    )