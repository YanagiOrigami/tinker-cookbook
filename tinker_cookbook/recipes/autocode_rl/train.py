"""
AutoCode RL — CLI entry point.

Trains a model on competitive programming tasks graded by a remote judge.
Training data is loaded from local JSONL files (``problem_id`` +
``problem_statement``).  The model receives an LCB-style prompt that allows
both Python and C++.

Two evaluation modes are supported:

- ``eval_mode=judge`` (default): evaluates on a local JSONL via the remote judge.
- ``eval_mode=lcb``: evaluates on LiveCodeBench via a local sandbox.

Example — toy debug run::

    python -m tinker_cookbook.recipes.autocode_rl.train \\
        model_name="meta-llama/Llama-3.2-1B" \\
        train_path="data/train.jsonl" \\
        test_path="data/eval.jsonl" \\
        group_size=2 groups_per_batch=4 max_tokens=512 max_steps=2

Example — full training with LCB eval::

    python -m tinker_cookbook.recipes.autocode_rl.train \\
        model_name="meta-llama/Llama-3.1-8B-Instruct" \\
        train_path="data/train.jsonl" \\
        eval_mode=lcb sandbox_backend=sandboxfusion \\
        group_size=4 groups_per_batch=100 learning_rate=1e-5 max_tokens=4096
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Literal

import chz
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.autocode_rl.autocode_env import AutocodeDatasetBuilder
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from tinker_cookbook.sandbox import SandboxBackend

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for AutoCode RL training.

    Environment variables:
        TINKER_API_KEY   — Tinker platform API key (required).
        JUDGE_HOST       — Remote judge hostname (default: ``localhost``).
        JUDGE_PORT       — Remote judge port (default: ``8081``).
        SANDBOX_URL      — SandboxFusion endpoint (only for ``eval_mode=lcb``).
    """

    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Data (local JSONL)
    train_path: str = "data/train.jsonl"
    test_path: str = "data/eval.jsonl"
    seed: int = 0

    # Evaluation mode: "judge" (eval.jsonl + remote judge) or "lcb" (LiveCodeBench + sandbox)
    eval_mode: Literal["judge", "lcb"] = "judge"
    sandbox_backend: SandboxBackend | None = None
    sandbox_timeout: int = 6

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 4096
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    format_coef: float = 0.1

    # Loss
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Dataset recurrence
    train_recurrent: int = 1

    # Logging / eval / checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False
    eval_every: int = 20
    save_every: int = 20

    # Service
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Async rollout
    max_steps_off_policy: int | None = None
    max_steps: int | None = None
    rollout_max_retries: int | None = None


async def cli_main(cli_config: CLIConfig) -> None:
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"autocode-{model_tag}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-seed{cli_config.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/autocode_rl/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    dataset_builder = AutocodeDatasetBuilder(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        train_path=cli_config.train_path,
        test_path=cli_config.test_path,
        eval_mode=cli_config.eval_mode,
        sandbox_backend=cli_config.sandbox_backend,
        sandbox_timeout=cli_config.sandbox_timeout,
        format_coef=cli_config.format_coef,
        seed=cli_config.seed,
        train_recurrent=cli_config.train_recurrent,
    )

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        max_steps=cli_config.max_steps,
        rollout_error_tolerance=RetryOnFailure(max_retries=cli_config.rollout_max_retries)
        if cli_config.rollout_max_retries is not None
        else False,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
