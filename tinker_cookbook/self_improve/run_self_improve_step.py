import os
import asyncio
import json
import random

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.coding_rl.coding_grading import remote_code_judge
from tinker_cookbook.checkpoint_utils import get_last_checkpoint
from tinker_cookbook.self_improve.generate_problems import _get_sampling_client_from_latest_checkpoint, _find_latest_run_dir, RL_LOG_ROOT

from tinker_cookbook.self_improve import generate_problems, build_tests_and_upload
from tinker_cookbook.recipes.coding_rl import train as rl_train


def run_loop() -> None:
    # x problems per epoch (defaults to 4), total epochs (defaults to 1000)
    x = int(os.environ.get("SELF_IMPROVE_NUM", "4"))
    num_epochs = int(os.environ.get("SELF_IMPROVE_EPOCHS", "1000"))
    base_dir = "/Users/zeyushen/Desktop/tinker-cookbook/data/self_improve"
    unsolved_path = os.path.join(base_dir, "unsolvable.jsonl")
    epoch_path = os.path.join(base_dir, "unsolved_epoch.jsonl")

    for epoch in range(1, num_epochs + 1):
        print(f"[self-improve] epoch {epoch}/{num_epochs} start; x={x}")

        # 1) Generate x new problems (added to unsolvable.jsonl)
        generate_problems.main()

        # 2) Build tests and upload to judge; also appends dataset.jsonl
        build_tests_and_upload.build_tests_and_upload()

        # 3) Select x problems from unsolvable set for this epoch and write epoch file
        unsolved = []
        if os.path.exists(unsolved_path):
            with open(unsolved_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        unsolved.append(json.loads(line))
                    except Exception:
                        continue
        random.shuffle(unsolved)
        pick = unsolved[:x]
        with open(epoch_path, "w", encoding="utf-8") as f:
            for r in pick:
                f.write(json.dumps({
                    "problem_id": r["problem_id"],
                    "problem_statement": r["problem_statement"],
                }) + "\n")

        # 4) Run a short RL step on selected unsolved problems (32 rollouts per problem)
        os.environ["SELF_IMPROVE_EPOCH_FILE"] = epoch_path
        # Load previous epoch checkpoint (continuous training)
        load_path = None
        try:
            last_run = _find_latest_run_dir(RL_LOG_ROOT)
            if last_run is not None:
                ckpt = get_last_checkpoint(last_run, required_key="state_path")
                if ckpt is not None:
                    load_path = ckpt["state_path"]
        except Exception:
            load_path = None

        cli_cfg = rl_train.CLIConfig(
            env="coding_generated",
            group_size=32,
            groups_per_batch=x,
            epochs=1,
            loss_fn="ppo",
            load_checkpoint_path=load_path,
        )
        asyncio.run(rl_train.cli_main(cli_cfg))

        # 5) Evaluate these x problems with current sampler and move solved ones
        tokenizer = get_tokenizer("openai/gpt-oss-120b")
        renderer_name = model_info.get_recommended_renderer_name("openai/gpt-oss-120b")
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        sampling_client = _get_sampling_client_from_latest_checkpoint()
        with open(epoch_path, "r", encoding="utf-8") as f:
            epoch_rows = [json.loads(line) for line in f if line.strip()]
        solved_ids: set[str] = set()
        for row in epoch_rows:
            pid = row["problem_id"]
            stmt = row["problem_statement"]
            convo = [
                {"role": "user", "content": stmt + "\n\n" +
                 "Please write the correct code to solve the above problem, remember to put code inside a code box like ```<lang>\n<your code>\n```. Do not output anything else after the code box. Your solution should be executable and correct."}
            ]
            model_input = renderer.build_generation_prompt(convo)
            sample_future = sampling_client.sample(
                prompt=model_input,
                num_samples=32,
                sampling_params=tinker.types.SamplingParams(max_tokens=25000, stop=renderer.get_stop_sequences()),
            )
            result = sample_future.result()
            num_correct = 0
            for seq in result.sequences:
                parsed, _ = renderer.parse_response(seq.tokens)
                content = parsed.get("content", "")
                if remote_code_judge(content, pid):
                    num_correct += 1
            if num_correct >= 16:
                solved_ids.add(pid)

        if solved_ids:
            solvable_path = os.path.join(base_dir, "solvable.jsonl")
            solvable_rows = []
            if os.path.exists(solvable_path):
                with open(solvable_path, "r", encoding="utf-8") as f:
                    solvable_rows = [json.loads(line) for line in f if line.strip()]
            unsolved_rows = []
            if os.path.exists(unsolved_path):
                with open(unsolved_path, "r", encoding="utf-8") as f:
                    unsolved_rows = [json.loads(line) for line in f if line.strip()]
            new_solvable = list(solvable_rows)
            pid_to_row = {r["problem_id"]: r for r in unsolved_rows}
            for pid in solved_ids:
                row = pid_to_row.get(pid)
                if row:
                    new_solvable.append({
                        "problem_id": row["problem_id"],
                        "problem_statement": row["problem_statement"],
                        "solution": "",
                    })
            with open(solvable_path, "w", encoding="utf-8") as f:
                for r in new_solvable:
                    f.write(json.dumps(r) + "\n")
            with open(unsolved_path, "w", encoding="utf-8") as f:
                for r in unsolved_rows:
                    if r.get("problem_id") not in solved_ids:
                        f.write(json.dumps(r) + "\n")
            print(f"[self-improve][epoch {epoch}] solved count: {len(new_solvable)}; unsolved remaining: {len([r for r in unsolved_rows if r.get('problem_id') not in solved_ids])}")


if __name__ == "__main__":
    run_loop()


