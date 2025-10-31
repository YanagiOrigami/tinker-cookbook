import os
import re
import csv
import json
import time
import random
from typing import Optional, Tuple, List
import httpx

from datasets import load_dataset

import tinker
from tinker import types as tinker_types
from tinker_cookbook import model_info
from tinker_cookbook.checkpoint_utils import get_last_checkpoint
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter

PARQUET_PATH = "/Users/zeyushen/Desktop/tinker-cookbook/LiveCodeBench-Pro/data/biannual_2024_7_12-00000-of-00001.parquet"
SOLUTION_CSV = "/Users/zeyushen/Desktop/tinker-cookbook/LiveCodeBench-Pro/data/solution.csv"
OUTPUT_DIR = "/Users/zeyushen/Desktop/tinker-cookbook/data/self_improve"
FAIL_DIR = os.path.join(OUTPUT_DIR, "failures")
# Sets
SOLVABLE_JSONL = os.path.join(OUTPUT_DIR, "solvable.jsonl")
UNSOLVABLE_JSONL = os.path.join(OUTPUT_DIR, "unsolvable.jsonl")
# Root where coding_rl logs live
RL_LOG_ROOT = "/tmp/tinker-examples/coding_rl"
# Problems per batch
NUM_PROBLEMS_PER_STEP = int(os.environ.get("SELF_IMPROVE_NUM", "4"))
# Base model used for tokenizer/renderer
BASE_MODEL_NAME = "openai/gpt-oss-120b"

FORMAT_SYSTEM_PROMPT = (
    "You are a strict formatter. Output EXACTLY two sections and nothing else.\n"
    "1) The new problem statement strictly between the tags:\n"
    "<begin of problem statement for the new problem>\n"
    "...\n"
    "<end of problem statement for the new problem>\n"
    "2) The accepted C++ solution strictly between the tags:\n"
    "<begin of accepted solution for the new problem>\n"
    "```cpp\n<full C++17 program with main()>\n```\n"
    "<end of accepted solution for the new problem>\n"
    "Do NOT include analysis, roles, extra commentary, or any text outside these two tagged sections."
)

PROMPT_TEMPLATE = (
    "This is a competitive programming problem statement and its accepted solution:\n\n"
    "<begin of problem statement>\n{problem_statement}\n<end of problem statement>\n\n"
    "<begin of solution>\n{accepted_solution}\n<end of solution>\n\n"
    "You are an experienced competitive programming problem setter. Please modify at least one condition of the problem to design a new, more difficult problem. "
    "This new problem must require a different solution approach from the original and should not be solvable using heuristic methods. The problem must be of such quality and novelty that it could be accepted on a platform like Codeforces, or at the very least serve as a valuable training exercise. "
    "In the problem statement, please include time and space constraints and provide sample test cases. Furthermore, please write an accepted C++ solution, along with a complexity analysis, a proof of correctness, and a clear explanation.\n\n"
    "Response Format:\n\n"
    "<begin of problem statement for the new problem>\n"
    "// Problem statement for the new problem\n"
    "<end of problem statement for the new problem>\n\n"
    "<begin of accepted solution for the new problem>\n"
    "// Accepted solution for the new problem\n"
    "<end of accepted solution for the new problem>\n\n"
    "Then, provide your solution explanation and proof.\n"
)

NEW_PROBLEM_STMT_RE = re.compile(
    r"<begin of problem statement for the new problem>\s*(.*?)\s*<end of problem statement for the new problem>",
    re.DOTALL,
)
NEW_SOLUTION_RE = re.compile(
    r"<begin of accepted solution for the new problem>\s*(.*?)\s*(?:<end of accepted solution for the new problem>|\Z)",
    re.DOTALL,
)


def _find_latest_run_dir(root: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subdirs:
        return None
    subdirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    return subdirs[0]


def _load_easy_problems(parquet_path: str) -> List[dict]:
    ds = load_dataset("parquet", data_files={"train": parquet_path})["train"]
    rows = [r for r in ds if str(r.get("difficulty", "")).lower() == "easy"]
    return rows


def _seed_solvable_if_missing() -> None:
    """Seed solvable.jsonl with all easy problems+solutions (if missing)."""
    if os.path.exists(SOLVABLE_JSONL):
        return
    ds = _load_easy_problems(PARQUET_PATH)
    sol_map = _load_solutions_map(SOLUTION_CSV)
    with open(SOLVABLE_JSONL, "w", encoding="utf-8") as f:
        for r in ds:
            pid = str(r.get("problem_id", "")).strip()
            stmt = str(r.get("problem_statement", "")).strip()
            sol = sol_map.get(pid)
            if not pid or not stmt or not sol:
                continue
            f.write(json.dumps({
                "problem_id": pid,
                "problem_statement": stmt,
                "solution": sol,
            }) + "\n")


def _load_solvable() -> List[dict]:
    if not os.path.exists(SOLVABLE_JSONL):
        _seed_solvable_if_missing()
    rows: List[dict] = []
    if os.path.exists(SOLVABLE_JSONL):
        with open(SOLVABLE_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def _load_solutions_map(csv_path: str) -> dict:
    sol_map: dict = {}
    # CSV without headers: first column problem_id, second column solution (may contain commas/newlines), quoted
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            pid = row[0].strip()
            solution = "".join(row[1:]) if len(row) > 1 else ""
            # Handle quoted content with csv module already; join() in case of extra commas splitted
            if pid and solution:
                sol_map[pid] = solution
    return sol_map


def _extract_new_problem_and_solution(text: str) -> Tuple[Optional[str], Optional[str]]:
    m_stmt = NEW_PROBLEM_STMT_RE.search(text)
    m_sol = NEW_SOLUTION_RE.search(text)
    new_stmt = m_stmt.group(1).strip() if m_stmt else None
    new_sol = m_sol.group(1).strip() if m_sol else None
    return new_stmt, new_sol


def _judge_problem_with_openrouter(problem_statement: str) -> tuple[bool, dict]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; cannot run LLM judge")
    url = "https://openrouter.ai/api/v1/chat/completions"
    system_prompt = (
        "You are a strict content validator for competitive programming. "
        "Determine if the USER content is a valid, non-placeholder, non-dummy problem statement suitable for an online judge. "
        "Reject content that is ellipses, templates, repeated filler, missing input/output specification, or generic instructions. "
        "Return ONLY a compact JSON object with fields: is_problem (boolean), reasons (array of strings)."
    )
    user_prompt = (
        "Problem statement to validate:\n\n" + problem_statement
    )
    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        # Force JSON-only output if supported
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Some providers return a list of segments; normalize to string
        if isinstance(content, list):
            parts: list[str] = []
            for seg in content:
                if isinstance(seg, dict):
                    parts.append(str(seg.get("text", "")))
                else:
                    parts.append(str(seg))
            content_str = "".join(parts).strip()
        else:
            content_str = str(content).strip()
        # Strip optional fenced code blocks
        if content_str.startswith("```"):
            # Attempt to extract the first fenced JSON block
            start = content_str.find("\n")
            end = content_str.rfind("```")
            if start != -1 and end != -1 and end > start:
                content_str = content_str[start + 1 : end].strip()
        # Try parse JSON
        result = json.loads(content_str)
        ok = bool(result.get("is_problem", False))
        return ok, {"raw": data, "parsed": result}
    except Exception as e:
        # Return rich debugging info to failure logs
        try:
            raw_text = resp.text  # type: ignore[misc]
        except Exception:
            raw_text = None
        return False, {"error": str(e), "raw_response_text": raw_text}


def _get_sampling_client_from_latest_checkpoint() -> tinker.SamplingClient:
    """Return latest sampler; if none exists (first epoch), use base model."""
    service_client = tinker.ServiceClient()
    run_dir = _find_latest_run_dir(RL_LOG_ROOT)
    if run_dir is None:
        # First epoch: no RL runs yet; use base model
        return service_client.create_sampling_client(base_model=BASE_MODEL_NAME)
    ckpt = get_last_checkpoint(run_dir, required_key="sampler_path")
    if ckpt is None:
        # No sampler checkpoint in the latest run; fall back to base model
        return service_client.create_sampling_client(base_model=BASE_MODEL_NAME)
    sampler_path = ckpt["sampler_path"]
    return service_client.create_sampling_client(model_path=sampler_path)


def _build_renderer() -> renderers.Renderer:
    tokenizer = get_tokenizer(BASE_MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)
    return renderers.get_renderer(renderer_name, tokenizer)


def _generate_with_tinker(prompt: str, sampling_client: tinker.SamplingClient, max_tokens: int = 28000) -> str:
    renderer = _build_renderer()
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)
    # Build a single-turn user message
    messages = [
        {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    model_input = renderer.build_generation_prompt(messages)
    sample_future = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker_types.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=["<end of accepted solution for the new problem>"]
        ),
    )
    sample_result = sample_future.result()
    tokens = sample_result.sequences[0].tokens
    parsed, _ = renderer.parse_response(tokens)
    return parsed.get("content", "")


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    _safe_mkdir(OUTPUT_DIR)
    _safe_mkdir(FAIL_DIR)

    # 1) Load solvable problems (seed from easy set if missing)
    solvable_rows = _load_solvable()

    # Prepare candidates with solution
    candidates: List[Tuple[str, str, str]] = []  # (problem_id, statement, solution)
    for r in solvable_rows:
        pid = str(r.get("problem_id", "")).strip()
        stmt = str(r.get("problem_statement", "")).strip()
        sol = str(r.get("solution", "")).strip()
        if pid and stmt and sol:
            candidates.append((pid, stmt, sol))

    if not candidates:
        raise RuntimeError("No easy problems with solutions found.")

    # Shuffle once; we'll cycle until we collect NUM_PROBLEMS_PER_STEP accepted
    random.shuffle(candidates)

    # 2) Use current Tinker sampler to generate new problems
    sampling_client = _get_sampling_client_from_latest_checkpoint()

    generated_ids: List[str] = []
    idx = 0
    attempts = 0
    max_attempts = 10_000_000  # effectively unlimited; we want exactly x accepted
    while len(generated_ids) < NUM_PROBLEMS_PER_STEP and attempts < max_attempts:
        orig_pid, stmt, sol = candidates[idx % len(candidates)]
        idx += 1
        attempts += 1
        prompt = PROMPT_TEMPLATE.format(problem_statement=stmt, accepted_solution=sol)
        new_stmt = None
        new_sol = None
        for attempt in range(3):
            content = _generate_with_tinker(prompt, sampling_client)
            # Save raw content for inspection
            ts_attempt = time.strftime("%Y%m%d%H%M%S")
            fail_path = os.path.join(FAIL_DIR, f"raw_{orig_pid}_{ts_attempt}_attempt{attempt+1}.txt")
            with open(fail_path, "w", encoding="utf-8") as f:
                f.write(content)
            new_stmt, new_sol = _extract_new_problem_and_solution(content)
            if new_stmt and new_sol:
                break
            else:
                print(f"Parse failed for {orig_pid} (attempt {attempt+1}/3), retrying...")
        if not new_stmt or not new_sol:
            print(f"Skipping {orig_pid}: failed to parse generated content after retries. See raw outputs under {FAIL_DIR}")
            continue
        # LLM judge to reject dummy/invalid problem statements
        ok, judge_info = _judge_problem_with_openrouter(new_stmt)
        if not ok:
            jpath = os.path.join(FAIL_DIR, f"judge_reject_{orig_pid}_{int(time.time())}.json")
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(judge_info, f, ensure_ascii=False, indent=2)
            print(f"Skipping {orig_pid}: LLM-judge rejected the generated problem. Details at {jpath}")
            continue
        ts = time.strftime("%Y%m%d%H%M%S")
        new_pid = f"SI_{orig_pid}_{ts}_{random.randint(1000,9999)}"
        out_dir = os.path.join(OUTPUT_DIR, new_pid)
        _safe_mkdir(out_dir)
        with open(os.path.join(out_dir, "statement.txt"), "w", encoding="utf-8") as f:
            f.write(new_stmt)
        # Save solution as C++ file
        with open(os.path.join(out_dir, "solution.cpp"), "w", encoding="utf-8") as f:
            f.write(new_sol)
        # Save the successful raw response as well
        with open(os.path.join(out_dir, "raw_response.txt"), "w", encoding="utf-8") as f:
            f.write(content)
        # Metadata
        meta = {"orig_pid": orig_pid, "new_pid": new_pid}
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        # Append to unsolvable set
        with open(UNSOLVABLE_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "problem_id": new_pid,
                "problem_statement": new_stmt,
                "generated_from": orig_pid,
            }) + "\n")
        generated_ids.append(new_pid)
        print(f"Generated new problem {new_pid} ({len(generated_ids)}/{NUM_PROBLEMS_PER_STEP})")

    # Save a manifest for next steps
    manifest = {
        "generated_ids": generated_ids,
        "output_dir": OUTPUT_DIR,
        "source_parquet": PARQUET_PATH,
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
