from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.autocode_rl.eval.benchmarks import (
    AIMEBenchmarkAdapter,
    GPQABenchmarkAdapter,
    GPQAConfigName,
    LCBBenchmarkAdapter,
    Math500BenchmarkAdapter,
)
from tinker_cookbook.autocode_rl.eval.common import (
    BenchmarkEvaluator,
    JSONValue,
    resolve_model_and_renderer_async,
    write_json,
    write_jsonl,
)
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.utils.ml_log import dump_config

logger = logging.getLogger(__name__)

BenchmarkName = Literal["lcb", "aime", "gpqa", "math500"]


@chz.chz
class CLIConfig:
    benchmark: BenchmarkName = "math500"

    model_name: str | None = None
    model_path: str | None = None
    renderer_name: str | None = None
    base_url: str | None = None

    max_examples: int | None = None
    max_parallel_tasks: int = 32
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 0

    output_dir: str | None = None
    behavior_if_output_dir_exists: cli_utils.LogdirBehavior = "ask"

    dataset_split: str | None = None
    use_standard_math_fewshot: bool = False
    math_grader: Literal["sympy", "math_verify"] = "sympy"
    math_grader_timeout: float = 1.0

    gpqa_config: GPQAConfigName = "gpqa_diamond"
    gpqa_shuffle_choices: bool = True

    sandbox_backend: SandboxBackend | None = None
    sandbox_timeout: int = 6


def _default_output_dir(benchmark: str, model_name: str) -> str:
    model_tag = model_name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"/tmp/tinker-examples/autocode_rl/eval/{benchmark}/{model_tag}-{timestamp}"


def _build_benchmark(config: CLIConfig):
    if config.benchmark == "math500":
        return Math500BenchmarkAdapter(
            split=config.dataset_split or "test",
            use_standard_fewshot=config.use_standard_math_fewshot,
            math_grader=config.math_grader,
            math_grader_timeout=config.math_grader_timeout,
        )
    if config.benchmark == "aime":
        return AIMEBenchmarkAdapter(
            split=config.dataset_split or "train",
            use_standard_fewshot=config.use_standard_math_fewshot,
            math_grader=config.math_grader,
            math_grader_timeout=config.math_grader_timeout,
        )
    if config.benchmark == "gpqa":
        return GPQABenchmarkAdapter(
            split=config.dataset_split or "train",
            gpqa_config=config.gpqa_config,
            shuffle_choices=config.gpqa_shuffle_choices,
            seed=config.seed,
        )
    if config.benchmark == "lcb":
        return LCBBenchmarkAdapter(
            split=cast(Literal["train", "test"], config.dataset_split or "test"),
            sandbox_backend=config.sandbox_backend,
            sandbox_timeout=config.sandbox_timeout,
            seed=config.seed,
        )
    raise AssertionError(f"Unsupported benchmark: {config.benchmark}")


async def cli_main(config: CLIConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
    )

    resolved = await resolve_model_and_renderer_async(
        model_name=config.model_name,
        renderer_name=config.renderer_name,
        model_path=config.model_path,
        base_url=config.base_url,
    )
    benchmark = _build_benchmark(config)
    output_dir = Path(config.output_dir or _default_output_dir(config.benchmark, resolved.model_name))
    cli_utils.check_log_dir(
        str(output_dir),
        behavior_if_exists=config.behavior_if_output_dir_exists,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = BenchmarkEvaluator(
        benchmark=benchmark,
        model_name=resolved.model_name,
        renderer_name=resolved.renderer_name,
        max_examples=config.max_examples,
        max_parallel_tasks=config.max_parallel_tasks,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
    )

    sampling_client = resolved.service_client.create_sampling_client(
        model_path=config.model_path,
        base_model=resolved.model_name,
    )

    started_at = perf_counter()
    summary, results = await evaluator.run(sampling_client)
    elapsed = round(perf_counter() - started_at, 2)

    resolved_config = dump_config(config)
    resolved_config["resolved_model_name"] = resolved.model_name
    resolved_config["resolved_renderer_name"] = resolved.renderer_name
    resolved_config["benchmark_details"] = benchmark.describe()

    summary_record: dict[str, JSONValue] = {
        "benchmark": config.benchmark,
        "model_name": resolved.model_name,
        "renderer_name": resolved.renderer_name,
        "model_path": config.model_path,
        "elapsed_seconds": elapsed,
        "output_dir": str(output_dir),
        **summary,
    }

    write_json(output_dir / "config.json", resolved_config)
    write_json(output_dir / "summary.json", summary_record)
    write_jsonl(output_dir / "predictions.jsonl", [result.to_record() for result in results])

    logger.info("Finished %s evaluation in %.2fs", config.benchmark, elapsed)
    for key, value in summary_record.items():
        logger.info("%s=%s", key, value)
    logger.info("Wrote summary to %s", output_dir / "summary.json")
    logger.info("Wrote per-example predictions to %s", output_dir / "predictions.jsonl")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
