from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import tinker

from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class EvalExample:
    index: int
    example_id: str
    messages: list[renderers.Message]
    payload: object | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class GradeResult:
    correct: bool
    extracted: bool
    prediction: str | None
    details: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ExampleResult:
    index: int
    example_id: str
    correct: bool
    extracted: bool
    prediction: str | None
    response_text: str
    output_tokens: int
    latency_sec: float
    response_parse_success: bool
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    details: dict[str, JSONValue] = field(default_factory=dict)
    error: str | None = None

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "index": self.index,
            "example_id": self.example_id,
            "correct": self.correct,
            "extracted": self.extracted,
            "prediction": self.prediction,
            "response_text": self.response_text,
            "output_tokens": self.output_tokens,
            "latency_sec": self.latency_sec,
            "response_parse_success": self.response_parse_success,
            "metadata": self.metadata,
            "details": self.details,
            "error": self.error,
        }


class BenchmarkAdapter(Protocol):
    name: str

    def load_examples(self, limit: int | None = None) -> list[EvalExample]:
        raise NotImplementedError

    async def grade_response(self, example: EvalExample, response_text: str) -> GradeResult:
        raise NotImplementedError

    def describe(self) -> dict[str, JSONValue]:
        raise NotImplementedError


@dataclass(frozen=True)
class ResolvedModelConfig:
    service_client: tinker.ServiceClient
    model_name: str
    renderer_name: str


async def resolve_model_and_renderer_async(
    *,
    model_name: str | None,
    renderer_name: str | None,
    model_path: str | None,
    base_url: str | None,
) -> ResolvedModelConfig:
    service_client = tinker.ServiceClient(base_url=base_url)
    resolved_model_name = model_name

    if model_path is not None:
        rest_client = service_client.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(model_path)
        if resolved_model_name is not None and resolved_model_name != training_run.base_model:
            raise ConfigurationError(
                f"model_name={resolved_model_name} does not match checkpoint base model "
                f"{training_run.base_model}"
            )
        resolved_model_name = resolved_model_name or training_run.base_model

    if resolved_model_name is None:
        raise ConfigurationError("model_name or model_path must be provided")

    resolved_renderer_name = (
        await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
            model_name=resolved_model_name,
            explicit_renderer_name=renderer_name,
            load_checkpoint_path=model_path,
            base_url=base_url,
        )
    )

    return ResolvedModelConfig(
        service_client=service_client,
        model_name=resolved_model_name,
        renderer_name=resolved_renderer_name,
    )


def write_json(path: Path, payload: object) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def write_jsonl(path: Path, records: list[dict[str, JSONValue]]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


class BenchmarkEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        *,
        benchmark: BenchmarkAdapter,
        model_name: str,
        renderer_name: str,
        max_examples: int | None,
        max_parallel_tasks: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        self.benchmark = benchmark
        self.examples = benchmark.load_examples(limit=max_examples)
        if not self.examples:
            raise ConfigurationError(f"No examples available for benchmark {benchmark.name}")

        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
        self.max_parallel_tasks = max_parallel_tasks
        self.sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=self.renderer.get_stop_sequences(),
        )

    async def _evaluate_one(
        self, example: EvalExample, sampling_client: tinker.SamplingClient
    ) -> ExampleResult:
        started_at = time.perf_counter()
        try:
            model_input = self.renderer.build_generation_prompt(example.messages)
            sample = await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=self.sampling_params,
            )
            sequence = sample.sequences[0]
            parsed_message, parse_success = self.renderer.parse_response(sequence.tokens)
            response_text = renderers.get_text_content(parsed_message)
            grade = await self.benchmark.grade_response(example, response_text)
            return ExampleResult(
                index=example.index,
                example_id=example.example_id,
                correct=grade.correct,
                extracted=grade.extracted,
                prediction=grade.prediction,
                response_text=response_text,
                output_tokens=len(sequence.tokens),
                latency_sec=round(time.perf_counter() - started_at, 4),
                response_parse_success=parse_success,
                metadata=example.metadata,
                details=grade.details,
            )
        except Exception as exc:
            logger.exception("Evaluation failed for %s", example.example_id)
            return ExampleResult(
                index=example.index,
                example_id=example.example_id,
                correct=False,
                extracted=False,
                prediction=None,
                response_text="",
                output_tokens=0,
                latency_sec=round(time.perf_counter() - started_at, 4),
                response_parse_success=False,
                metadata=example.metadata,
                details={},
                error=str(exc),
            )

    async def run(
        self, sampling_client: tinker.SamplingClient
    ) -> tuple[dict[str, JSONValue], list[ExampleResult]]:
        semaphore = asyncio.Semaphore(self.max_parallel_tasks)

        async def bounded_eval(example: EvalExample) -> ExampleResult:
            async with semaphore:
                return await self._evaluate_one(example, sampling_client)

        tasks = [asyncio.create_task(bounded_eval(example)) for example in self.examples]
        results: list[ExampleResult] = []

        for idx, task in enumerate(asyncio.as_completed(tasks), start=1):
            results.append(await task)
            if idx % 10 == 0 or idx == len(tasks):
                logger.info(
                    "Completed %d/%d examples for %s",
                    idx,
                    len(tasks),
                    self.benchmark.name,
                )

        results.sort(key=lambda item: item.index)
        return self._summarize(results), results

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        summary, _results = await self.run(sampling_client)
        return {
            key: float(value)
            for key, value in summary.items()
            if isinstance(value, (float, int)) and "/" in key
        }

    def _summarize(self, results: list[ExampleResult]) -> dict[str, JSONValue]:
        num_examples = len(results)
        num_correct = sum(int(result.correct) for result in results)
        num_extracted = sum(int(result.extracted) for result in results)
        num_parse_success = sum(int(result.response_parse_success) for result in results)
        num_errors = sum(int(result.error is not None) for result in results)
        prefix = self.benchmark.name

        return {
            "benchmark": prefix,
            "num_examples": num_examples,
            "num_correct": num_correct,
            f"{prefix}/accuracy": num_correct / num_examples,
            f"{prefix}/extract_rate": num_extracted / num_examples,
            f"{prefix}/response_parse_rate": num_parse_success / num_examples,
            f"{prefix}/error_rate": num_errors / num_examples,
            f"{prefix}/avg_output_tokens": sum(result.output_tokens for result in results)
            / num_examples,
            f"{prefix}/avg_latency_sec": sum(result.latency_sec for result in results)
            / num_examples,
        }
