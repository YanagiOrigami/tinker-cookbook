# Autocode RL Eval

这里放的是一套独立于 `recipes/` 的 benchmark eval 程序，直接用 Tinker 的 `SamplingClient` 跑推理，然后按 benchmark 类型做本地判分。

目标是两件事：

1. 尽量复用仓库里已经有的能力，而不是重新造一套。
2. 给你一个统一入口，再配四个薄 wrapper，日常用起来直接。

## 复用关系

- `LCB`：直接复用 [`tinker_cookbook/recipes/autocode_rl/autocode_env.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/recipes/autocode_rl/autocode_env.py) 里的 `load_lcb_tasks`，以及 [`tinker_cookbook/recipes/code_rl/code_grading.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/recipes/code_rl/code_grading.py) 里的 code extraction + sandbox grading。
- `AIME / Math500`：复用 [`tinker_cookbook/recipes/math_rl/math_env.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/recipes/math_rl/math_env.py) 的 grading 入口，以及 [`tinker_cookbook/recipes/math_rl/math_grading.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/recipes/math_rl/math_grading.py) 的 `extract_boxed` / `grade_answer` / `math_verify` 兼容逻辑。
- 模型 / renderer 解析：复用 [`tinker_cookbook/checkpoint_utils.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/checkpoint_utils.py)，所以可以直接评 base model，也可以评 Tinker checkpoint。

## 文件说明

- [`common.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/autocode_rl/eval/common.py)：共享 evaluator、并发采样、结果汇总、结果落盘。
- [`benchmarks.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/autocode_rl/eval/benchmarks.py)：四个 benchmark 的 dataset loader、prompt、answer extraction、grader。
- [`run.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/autocode_rl/eval/run.py)：统一 CLI。
- [`eval_math500.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/autocode_rl/eval/eval_math500.py)：Math500 专用入口。
- [`eval_aime.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/autocode_rl/eval/eval_aime.py)：AIME 专用入口。
- [`eval_gpqa.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/autocode_rl/eval/eval_gpqa.py)：GPQA 专用入口。
- [`eval_lcb.py`](/home/yanagi_origami/tinker-cookbook/tinker_cookbook/autocode_rl/eval/eval_lcb.py)：LiveCodeBench 专用入口。

## 安装

基础依赖：

```bash
uv pip install -e .
```

如果要跑 `AIME` 或 `Math500`，还需要 math grading 依赖：

```bash
uv pip install -e '.[math-rl]'
```

如果要跑 `GPQA`，数据集是 gated 的。你需要：

1. 去 Hugging Face 接受 `Idavidrein/gpqa` 的使用条款。
2. 本地登录 HF，或者设置 `HF_TOKEN`。

例如：

```bash
huggingface-cli login
```

如果要跑 `LCB`，需要本地 sandbox。最简单的是跑 SandboxFusion：

```bash
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
export SANDBOX_URL=http://localhost:8080/run_code
```

## 环境变量

至少需要：

```bash
export TINKER_API_KEY=...
```

可选：

```bash
export HF_TOKEN=...   # GPQA gated dataset 常用
export SANDBOX_URL=...  # LCB + sandboxfusion 时使用
```

## 最常用的运行方式

### 1. 统一入口

```bash
python -m tinker_cookbook.autocode_rl.eval.run \
    benchmark=math500 \
    model_name="Qwen/Qwen3-8B" \
    max_examples=50 \
    max_tokens=1024 \
    behavior_if_output_dir_exists=delete
```

支持的 `benchmark`：

- `math500`
- `aime`
- `gpqa`
- `lcb`

### 2. 专用入口

这几个 wrapper 本质上只是把 `benchmark=...` 固定住了。

Math500:

```bash
python -m tinker_cookbook.autocode_rl.eval.eval_math500 \
    model_name="Qwen/Qwen3-8B" \
    max_examples=100 \
    max_tokens=1024 \
    behavior_if_output_dir_exists=delete
```

AIME:

```bash
python -m tinker_cookbook.autocode_rl.eval.eval_aime \
    model_name="Qwen/Qwen3-8B" \
    max_tokens=1024 \
    behavior_if_output_dir_exists=delete
```

GPQA:

```bash
python -m tinker_cookbook.autocode_rl.eval.eval_gpqa \
    model_name="meta-llama/Llama-3.1-8B-Instruct" \
    gpqa_config=gpqa_diamond \
    max_examples=100 \
    max_tokens=512 \
    behavior_if_output_dir_exists=delete
```

LCB:

```bash
python -m tinker_cookbook.autocode_rl.eval.eval_lcb \
    model_name="Qwen/Qwen3-8B" \
    sandbox_backend=sandboxfusion \
    max_examples=25 \
    max_tokens=4096 \
    behavior_if_output_dir_exists=delete
```

## 评 checkpoint

如果你想评 Tinker 训练出来的 checkpoint，直接传 `model_path` 就行。

```bash
python -m tinker_cookbook.autocode_rl.eval.eval_math500 \
    model_path="tinker://<run-id>/sampler_weights/final" \
    max_examples=100 \
    max_tokens=1024 \
    behavior_if_output_dir_exists=delete
```

这时：

- `model_name` 可以不传，程序会从 checkpoint 对应的 training run 自动解析 base model。
- `renderer_name` 也可以不传，程序会优先从 checkpoint metadata 里取；没有的话再退回推荐 renderer。

## 关键参数

所有入口共用同一套 CLI 参数。

### 模型相关

- `model_name`: 直接评 base model 时使用。
- `model_path`: 评 Tinker checkpoint 时使用。
- `renderer_name`: 手动指定 renderer；默认自动推断。
- `base_url`: 自定义 Tinker service endpoint。

### 采样相关

- `max_examples`: 只评前 N 条，适合 smoke test。
- `max_parallel_tasks`: 并发请求数。`LCB` 建议保守一点，比如 `8~32`。
- `max_tokens`: 每题最大生成 token 数。
- `temperature`: 默认 `0.0`，适合 benchmark eval。
- `top_p`
- `top_k`

### 输出相关

- `output_dir`: 手动指定结果目录。
- `behavior_if_output_dir_exists`: `delete | resume | ask | raise`

### 数据集 / benchmark 相关

- `dataset_split`: 手动覆盖默认 split。
- `gpqa_config`: `gpqa_diamond | gpqa_main | gpqa_extended | gpqa_experts`
- `gpqa_shuffle_choices`: 是否对选项做稳定重排，默认 `True`
- `sandbox_backend`: `sandboxfusion | modal`
- `sandbox_timeout`: LCB 单题 sandbox 超时
- `use_standard_math_fewshot`: 是否给 `AIME / Math500` 加上 math_rl 里的标准 fewshot，默认 `False`
- `math_grader`: `sympy | math_verify`
- `math_grader_timeout`: 数学答案判分超时

## 默认 split

- `Math500`: `test`
- `AIME`: `train`
- `GPQA`: `train`
- `LCB`: `test`

如果你想改，直接传 `dataset_split=...`。

## 输出文件

每次运行会写到一个新的目录，默认类似：

```text
/tmp/tinker-examples/autocode_rl/eval/<benchmark>/<model>-<timestamp>/
```

里面主要有：

- `config.json`: 本次运行参数，外加解析后的 `resolved_model_name` / `resolved_renderer_name`
- `summary.json`: 聚合指标
- `predictions.jsonl`: 每题结果

`summary.json` 里会有这些字段：

- `num_examples`
- `num_correct`
- `<benchmark>/accuracy`
- `<benchmark>/extract_rate`
- `<benchmark>/response_parse_rate`
- `<benchmark>/error_rate`
- `<benchmark>/avg_output_tokens`
- `<benchmark>/avg_latency_sec`

`predictions.jsonl` 每行会包含：

- `example_id`
- `correct`
- `prediction`
- `response_text`
- `output_tokens`
- `latency_sec`
- `metadata`
- `details`
- `error`

## Benchmark 细节说明

### Math500

- 数据集：`HuggingFaceH4/MATH-500`
- prompt：在原题后面追加 `Write your answer in \boxed{} format.`
- 判分：复用 `math_rl` 的 `safe_grade`

### AIME

- 数据集：`HuggingFaceH4/aime_2024`
- prompt：额外强调答案是 `0..999` 的整数，并要求 `\boxed{}`
- 判分：同样复用 `math_rl` 的 grader

### GPQA

- 数据集：`Idavidrein/gpqa`
- 默认 config：`gpqa_diamond`
- prompt：把四个选项整理成 `A/B/C/D`，要求最后给一个 boxed letter
- 判分：提取 `A-D` 最终答案并比较

### LCB

- 数据集加载：直接复用 `recipes/autocode_rl/autocode_env.py` 里的 `load_lcb_tasks`
- prompt：沿用现有 LCB prompt
- 判分：提取 fenced code block，然后走现有 sandbox correctness check

## 建议的 smoke test

先别上来就全量跑，先用很小的 `max_examples` 验证链路。

Math500:

```bash
python -m tinker_cookbook.autocode_rl.eval.eval_math500 \
    model_name="meta-llama/Llama-3.2-1B" \
    max_examples=5 \
    max_tokens=512 \
    behavior_if_output_dir_exists=delete
```

GPQA:

```bash
python -m tinker_cookbook.autocode_rl.eval.eval_gpqa \
    model_name="meta-llama/Llama-3.2-1B" \
    max_examples=5 \
    max_tokens=256 \
    behavior_if_output_dir_exists=delete
```

LCB:

```bash
python -m tinker_cookbook.autocode_rl.eval.eval_lcb \
    model_name="meta-llama/Llama-3.2-1B" \
    sandbox_backend=sandboxfusion \
    max_examples=3 \
    max_parallel_tasks=2 \
    max_tokens=1024 \
    behavior_if_output_dir_exists=delete
```

## 几个实际建议

1. `LCB` 不要把并发一开始就开很大。瓶颈通常在 sandbox，不在 Tinker。
2. `GPQA` 默认我做了稳定选项重排，避免模型吃固定位置偏置。
3. `AIME / Math500` 默认是 zero-shot；如果你要贴近你在 `math_rl` 里的风格，可以开 `use_standard_math_fewshot=True`。
4. 这些脚本是“实用型 Tinker-native eval runner”，不是逐 token 复刻某个 leaderboard 的官方 prompt stack。如果你要和外部榜单严格对齐，通常要继续固定 prompt、stop condition、fewshot、choice order 和 grader 细节。
