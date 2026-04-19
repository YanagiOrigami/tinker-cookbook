# AutoCode RL — Competitive Programming with Remote Judge

This recipe trains language models on competitive programming problems using RL. Unlike `code_rl` (which runs code in a local Docker sandbox), this recipe submits solutions to an external judge server and receives pass/fail verdicts.

## Architecture

```
Model  ──>  Generate code  ──>  Extract code + detect language  ──>  POST /submit
                                                                         │
                                                 Reward (0 / 1)  <───────┘
```

- **Single-turn**: the model sees a problem statement and produces a fenced code block.
- **Remote judge**: code is sent via HTTP to a configurable judge server. The judge compiles, runs, and returns pass/fail.
- **Multi-language**: C++ and Python are supported (auto-detected from the code block tag).

## Prompt Design

The model receives a prompt modeled after LiveCodeBench but explicitly allowing **Python or C++**:

```
You are an expert programmer. You will be given a question (problem specification)
and will generate a correct program that matches the specification and passes all
tests. You may write your solution in either Python or C++.

<problem statement>

### Format: Read the inputs from stdin, solve the problem, and write the answer
to stdout (do not directly test on the sample inputs). Enclose your code within
delimiters as follows.
```python
# YOUR CODE HERE
```
or
```cpp
// YOUR CODE HERE
```

### Answer: (use the provided format with backticks)
```

A one-shot fewshot prefix (a simple stdin/stdout C++ example) is included by default. Set `convo_prefix=None` in the builder to disable it.

## Data Format

Training and evaluation data are local **JSONL** files. Each line:

```json
{"problem_id": "1234A", "problem_statement": "Given an integer n, print ..."}
```

- `problem_id` — identifier the judge uses to select the correct test cases.
- `problem_statement` — full problem text shown to the model.

## Setup

### 1. API Key

Sign up for [Tinker](https://auth.thinkingmachines.ai/sign-up), create an API key from the [console](https://tinker-console.thinkingmachines.ai), and export it:

```bash
export TINKER_API_KEY="your-key-here"
```

### 2. Judge Server

The judge must be reachable at `http://<JUDGE_HOST>:<JUDGE_PORT>` and implement:

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/submit` | POST | `{"pid": str, "lang": str, "code": str}` | `{"sid": int}` |
| `/result/{sid}` | GET | — | `{"status": "done"\|"pending", "passed": bool}` |

Defaults (configurable via environment variables):

```bash
export JUDGE_HOST=localhost   # default
export JUDGE_PORT=8081        # default
```

### 3. Prepare Data

Place your JSONL files at the desired paths. Defaults:

```
data/train.jsonl
data/eval.jsonl
```

Override with `train_path=...` and `test_path=...`.

## Training

```bash
python -m tinker_cookbook.recipes.autocode_rl.train \
    model_name="meta-llama/Llama-3.1-8B-Instruct" \
    train_path="data/train.jsonl" \
    test_path="data/eval.jsonl" \
    group_size=4 \
    groups_per_batch=100 \
    learning_rate=1e-5 \
    max_tokens=4096
```

### Toy Debug Run

Use a smaller model and minimal batch to verify the pipeline end-to-end:

```bash
python -m tinker_cookbook.recipes.autocode_rl.train \
    model_name="meta-llama/Llama-3.2-1B" \
    train_path="data/train.jsonl" \
    test_path="data/eval.jsonl" \
    group_size=2 \
    groups_per_batch=4 \
    max_tokens=512 \
    max_steps=2 \
    eval_every=1 \
    behavior_if_log_dir_exists=delete
```

## Evaluation Modes

Two evaluation modes are available, controlled by `eval_mode`:

### `eval_mode=judge` (default)

Evaluates on `eval.jsonl` using the same remote judge as training. No extra setup needed.

### `eval_mode=lcb`

Evaluates on **LiveCodeBench** (loaded from HuggingFace `agentica-org/DeepCoder-Preview-Dataset`). Code is executed in a local sandbox (SandboxFusion or Modal) — the same pipeline as `code_rl`. Note: LCB eval uses the standard **Python-only** LCB prompt, not the autocode multi-language prompt.

This requires a running sandbox. Start one with:

```bash
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```

Then run training with LCB eval:

```bash
python -m tinker_cookbook.recipes.autocode_rl.train \
    model_name="meta-llama/Llama-3.1-8B-Instruct" \
    eval_mode=lcb \
    sandbox_backend=sandboxfusion \
    train_path="data/train.jsonl" \
    group_size=4 \
    groups_per_batch=100 \
    learning_rate=1e-5 \
    max_tokens=4096
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `meta-llama/Llama-3.1-8B-Instruct` | Base model |
| `train_path` | `data/train.jsonl` | Training JSONL path |
| `test_path` | `data/eval.jsonl` | Eval JSONL path (used when `eval_mode=judge`) |
| `eval_mode` | `judge` | `"judge"` (remote judge) or `"lcb"` (LiveCodeBench sandbox) |
| `sandbox_backend` | `None` | `"sandboxfusion"` or `"modal"` (only for `eval_mode=lcb`) |
| `sandbox_timeout` | 6 | Per-test timeout in seconds (only for `eval_mode=lcb`) |
| `group_size` | 4 | Rollouts per problem |
| `groups_per_batch` | 100 | Problems per training batch |
| `learning_rate` | 1e-5 | Learning rate |
| `max_tokens` | 4096 | Max generation tokens |
| `lora_rank` | 32 | LoRA rank |
| `format_coef` | 0.1 | Weight for format reward penalty |
| `eval_every` | 20 | Steps between evaluations |
| `save_every` | 20 | Steps between checkpoints |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TINKER_API_KEY` | — | Tinker platform API key (required) |
| `JUDGE_HOST` | `localhost` | Remote judge hostname |
| `JUDGE_PORT` | `8081` | Remote judge port |
| `SANDBOX_URL` | `http://localhost:8080/run_code` | SandboxFusion endpoint (only for `eval_mode=lcb`) |

## File Structure

```
autocode_rl/
├── autocode_env.py       # AutocodeEnv (remote judge) + LCBEvalEnv (sandbox), datasets, builders
├── autocode_grading.py   # Code extraction, language detection, remote judge HTTP client
├── train.py              # CLI entry point with CLIConfig
├── lcb_utils.py          # LiveCodeBench test runner utilities (from code_rl)
└── README.md
```
