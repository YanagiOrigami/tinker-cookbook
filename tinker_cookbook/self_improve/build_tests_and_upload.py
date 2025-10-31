import os
import re
import json
import shutil
import subprocess
import zipfile
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp
import asyncio

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.completers import TinkerTokenCompleter


BASE_MODEL_NAME = "openai/gpt-oss-120b"
OUTPUT_DIR = "/Users/zeyushen/Desktop/tinker-cookbook/data/self_improve"
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.json")
JUDGE_BASE_URL = "http://38.80.122.117:8081"
TESTLIB_PATH = "/Users/zeyushen/Desktop/testcase_generation/lib/testlib.h"
NUM_TESTCASES = 10
NUM_GENERATOR_CANDIDATES = 3
NUM_VALIDATOR_CANDIDATES = 2
NUM_CHECKER_CANDIDATES = 2

# Templates from your testcase_generation repo (gg3 + validator/checker)
GEN_TEMPLATE_PATH = "/Users/zeyushen/Desktop/testcase_generation/src/generator/generator_template3.txt"
VAL_TEMPLATE_PATH = "/Users/zeyushen/Desktop/testcase_generation/src/validator/validator_template.txt"
CHK_TEMPLATE_PATH = "/Users/zeyushen/Desktop/testcase_generation/src/checker/checker_template.txt"
VAL_TESTS_TEMPLATE_PATH = "/Users/zeyushen/Desktop/testcase_generation/src/validator/validator_tests_template.txt"
CHK_TESTS_TEMPLATE_PATH = "/Users/zeyushen/Desktop/testcase_generation/src/checker/checker_tests_template.txt"
STD_CHECKERS_DIR = "/Users/zeyushen/Desktop/testcase_generation/src/checker/standard_checkers"

# Verbatim format instructions extracted from templates (format-only, no extra guidance)
VAL_TESTS_FORMAT_SYS = (
    "Write 10 validator tests for this problem.\n\n"
    "Your response should contain the validator tests enclosed in a single markdown code block.\n\n"
    "Use \"verdict\": 1 when the input meets the problem constraints, and \"verdict\": 0 otherwise.\n\n"
    "Provide 3 valid inputs and 7 invalid inputs.\n\n"
    "The format should look like this:\n"
    "```json\n"
    "[\n"
    "  { \"input\": \"...\", \"verdict\": 1 },\n"
    "  { \"input\": \"...\", \"verdict\": 0 },\n"
    "  ...\n"
    "]\n"
    "```"
)

CHK_TESTS_FORMAT_SYS = (
    "Write 10 checker tests for this problem.\n\n"
    "Your response should contain the checker tests enclosed in a single markdown code block.\n\n"
    "Use \"verdict\": 1 when the output meets the problem constraints, and \"verdict\": 0 otherwise.\n\n"
    "Provide 3 correct outputs and 7 incorrect outputs.\n\n"
    "input is valid input, output is the contestant's output, answer is the jury's output, and verdict indicates whether the output meets the problem's constraints.\n\n"
    "The format should look like this:\n"
    "```json\n"
    "[\n"
    "  { \"input\": \"...\", \"output\": \"...\", \"answer\": \"...\", \"verdict\": 1 },\n"
    "  { \"input\": \"...\", \"output\": \"...\", \"answer\": \"...\", \"verdict\": 0 },\n"
    "  ...\n"
    "]\n"
    "```"
)

GEN_FORMAT_SYS = (
    "Your response should contain the generator enclosed in a single markdown block:\n"
    "```cpp\n"
    "//generator code here\n"
    "```"
)

VAL_FORMAT_SYS = (
    "Output only a single fenced C++ code block containing the validator implementation:\n"
    "```cpp\n"
    "//validator code here\n"
    "```"
)

CHK_FORMAT_SYS = (
    "If a standard checker is sufficient, your response should contain only the name of the appropriate .cpp file from the list below.\n"
    "fcmp.cpp: Compares files line by line, doesn't ignore trailing whitespaces on each line.\n"
    "lcmp.cpp: Compares files line by line, ignoring trailing whitespace on each line.\n"
    "hcmp.cpp: Compares a single huge integer.\n"
    "wcmp.cpp: Compares sequences of tokens, ignoring trailing whitespace.\n"
    "ncmp.cpp: Compares sequences of integers, ignoring trailing whitespace.\n"
    "rcmp4.cpp: Compares sequences of real numbers with a precision of 1E-4.\n"
    "rcmp6.cpp: Compares sequences of real numbers with a precision of 1E-6.\n"
    "rcmp9.cpp: Compares sequences of real numbers with a precision of 1E-9.\n"
    "yesno.cpp: Compares a single \"yes\" or \"no\" token, case-insensitively.\n"
    "For example, reply \"wcmp.cpp\", do not include quotation marks and other content.\n\n"
    "If the problem requires custom validation logic that a checker comparator cannot handle, your response should contain the checker enclosed in a single markdown block:\n"
    "```cpp\n"
    "//checker code here\n"
    "```"
)


def _build_renderer() -> renderers.Renderer:
    tokenizer = get_tokenizer(BASE_MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)
    return renderers.get_renderer(renderer_name, tokenizer)


def _get_latest_sampler_client() -> tinker.SamplingClient:
    """Return latest sampler; if none exists (first epoch), use base model."""
    rl_root = "/tmp/tinker-examples/coding_rl"
    service_client = tinker.ServiceClient()
    try:
        subdirs = [os.path.join(rl_root, d) for d in os.listdir(rl_root) if os.path.isdir(os.path.join(rl_root, d))]
    except FileNotFoundError:
        subdirs = []
    if not subdirs:
        return service_client.create_sampling_client(base_model=BASE_MODEL_NAME)
    subdirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    run_dir = subdirs[0]
    ckpt_file = os.path.join(run_dir, "checkpoints.jsonl")
    if not os.path.exists(ckpt_file):
        return service_client.create_sampling_client(base_model=BASE_MODEL_NAME)
    last_sampler = None
    with open(ckpt_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if "sampler_path" in row:
                    last_sampler = row["sampler_path"]
            except Exception:
                continue
    if not last_sampler:
        return service_client.create_sampling_client(base_model=BASE_MODEL_NAME)
    return service_client.create_sampling_client(model_path=last_sampler)


def _sample_tinker(prompt: str, sampling_client: tinker.SamplingClient, max_tokens, system_prompt: str | None = None) -> str:
    renderer = _build_renderer()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    model_input = renderer.build_generation_prompt(messages)
    result = sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=tinker.types.SamplingParams(max_tokens=max_tokens)).result()
    tokens = result.sequences[0].tokens
    parsed, _ = renderer.parse_response(tokens)
    return parsed.get("content", "")


def _extract_cpp(md: str) -> Optional[str]:
    m = re.search(r"```cpp\s+(.*?)```", md, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```\s+(.*?)```", md, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    # Heuristic: accept unfenced C++ if it looks like a complete program
    text = md.strip()
    if ("int main" in text) and ("#include" in text or "testlib.h" in text):
        return text
    return None

def _gen_cpp_with_retries(prompt: str, sampling_client: tinker.SamplingClient, role: str, max_tokens: int = 28000) -> str:
    suffix = (
        "\n\nReturn only a complete C++17 program with a main function in a single fenced code block:\n"
        "```cpp\n<code>\n```\n"
        "Do not include any explanations or extra text."
    )
    for attempt in range(3):
        md = _sample_tinker(prompt + suffix, sampling_client, max_tokens=max_tokens)
        cpp = _extract_cpp(md)
        if cpp and "int main" in cpp:
            return cpp
    raise RuntimeError(f"Failed to generate valid {role} after retries")


def _require_json_from_template(template_path: str, statement: str, solution: str | None, sampling_client: tinker.SamplingClient, max_tokens: int = 8000) -> list[dict]:
    tmpl = _load_template(template_path)
    prompt = tmpl.replace("{{problem statement}}", statement)
    if solution is not None:
        prompt = prompt.replace("{{accepted solution}}", solution)
    # Use only the format instructions verbatim in the system prompt
    if os.path.basename(template_path) == os.path.basename(VAL_TESTS_TEMPLATE_PATH):
        sys = VAL_TESTS_FORMAT_SYS
    elif os.path.basename(template_path) == os.path.basename(CHK_TESTS_TEMPLATE_PATH):
        sys = CHK_TESTS_FORMAT_SYS
    else:
        sys = ""
    renderer = _build_renderer()
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": prompt},
    ]
    model_input = renderer.build_generation_prompt(messages)
    t0 = time.time()
    for attempt in range(3):
        print(f"[llm] request from template {os.path.basename(template_path)} attempt {attempt+1}/3", flush=True)
        result = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.types.SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0, stop=[]),
        ).result()
        tokens = result.sequences[0].tokens
        content, _ = renderer.parse_response(tokens)
        md = content.get("content", "")
        m = re.search(r"```json\s*(.*?)```", md, re.DOTALL)
        payload = m.group(1) if m else md.strip()
        try:
            data = json.loads(payload)
            if isinstance(data, list):
                print(f"[llm] received valid JSON ({len(data)} items) in {time.time()-t0:.2f}s", flush=True)
                return data
        except Exception:
            pass
    raise RuntimeError("Failed to get valid JSON from template after retries")


def _generate_validator_tests(pdir: str, statement: str, sampling_client: tinker.SamplingClient) -> list[tuple[str, str]]:
    print("[tests] generating validator tests", flush=True)
    t0 = time.time()
    tests = _require_json_from_template(VAL_TESTS_TEMPLATE_PATH, statement, None, sampling_client)
    out_dir = os.path.join(pdir, "tests", "validator")
    os.makedirs(out_dir, exist_ok=True)
    paths: list[tuple[str, str]] = []
    idx = 1
    for t in tests[:NUM_TESTCASES]:
        inp = str(t.get("input", ""))
        verdict = str(t.get("verdict", "0"))
        in_path = os.path.join(out_dir, f"{idx}.in")
        res_path = os.path.join(out_dir, f"{idx}.res")
        _write_file(in_path, inp)
        _write_file(res_path, verdict)
        paths.append((in_path, res_path))
        idx += 1
    print(f"[tests] validator tests ready: {len(paths)} in {time.time()-t0:.2f}s", flush=True)
    return paths


def _generate_checker_tests(pdir: str, statement: str, solution: str, sampling_client: tinker.SamplingClient) -> list[tuple[str, str, str, str]]:
    print("[tests] generating checker tests", flush=True)
    t0 = time.time()
    tests = _require_json_from_template(CHK_TESTS_TEMPLATE_PATH, statement, solution, sampling_client)
    out_dir = os.path.join(pdir, "tests", "checker")
    os.makedirs(out_dir, exist_ok=True)
    paths: list[tuple[str, str, str, str]] = []
    idx = 1
    for t in tests[:NUM_TESTCASES]:
        inp = str(t.get("input", ""))
        out = str(t.get("output", ""))
        ans = str(t.get("answer", ""))
        verdict = str(t.get("verdict", "0"))
        in_path = os.path.join(out_dir, f"{idx}.in")
        out_path = os.path.join(out_dir, f"{idx}.out")
        ans_path = os.path.join(out_dir, f"{idx}.ans")
        res_path = os.path.join(out_dir, f"{idx}.res")
        _write_file(in_path, inp)
        _write_file(out_path, out)
        _write_file(ans_path, ans)
        _write_file(res_path, verdict)
        paths.append((in_path, out_path, ans_path, res_path))
        idx += 1
    print(f"[tests] checker tests ready: {len(paths)} in {time.time()-t0:.2f}s", flush=True)
    return paths


def _generate_candidates_from_template(tmpl_path: str, statement: str, solution: str | None, role: str, sampling_client: tinker.SamplingClient, num: int) -> list[str]:
    print(f"[candidates] generating {num} {role}(s) using {os.path.basename(tmpl_path)}", flush=True)
    tmpl = _load_template(tmpl_path)
    prompt = tmpl.replace("{{problem statement}}", statement)
    if solution is not None:
        prompt = prompt.replace("{{accepted solution}}", solution)
    # Use only format instructions as system prompt (verbatim from templates)
    if role == "generator":
        sys = GEN_FORMAT_SYS
    elif role == "validator":
        sys = VAL_FORMAT_SYS
    elif role == "checker":
        sys = CHK_FORMAT_SYS
    else:
        sys = ""
    codes: list[str] = []
    for _ in range(num):
        # Checker template may return a standard file name; handle below in caller
        _head = prompt[:160].replace("\n", " ")
        print(f"[debug] {role} prompt head: {_head}...", flush=True)
        md = _sample_tinker(prompt, sampling_client, max_tokens=20000, system_prompt=sys)
        code = _extract_cpp(md)
        if code:
            codes.append(code)
        else:
            # Possibly a standard checker name
            text = md.strip()
            codes.append(text)
    return codes


def _score_validators(pdir: str, candidates: list[str], val_tests: list[tuple[str, str]]) -> tuple[int, str]:
    print(f"[score] validators: {len(candidates)} candidates, {len(val_tests)} tests", flush=True)
    best_idx = -1
    best_score = -1
    for i, code in enumerate(candidates):
        src = os.path.join(pdir, f"validator_candidate_{i}.cpp")
        _write_file(src, code)
        try:
            bin_path = os.path.join(pdir, f"validator_candidate_{i}.out")
            _compile_cpp(src, bin_path)
        except Exception as e:
            print(f"[validator candidate {i}] compile error: {e}")
            continue
        score = 0
        # parallel per-test runs
        def run_one(test_pair: tuple[str, str]) -> int:
            in_path, res_path = test_pair
            try:
                with open(in_path, "r", encoding="utf-8") as fin:
                    res = subprocess.run([bin_path], stdin=fin, capture_output=True, text=True, timeout=10)
                expected = open(res_path, "r", encoding="utf-8").read().strip()
                should_be_valid = (expected == "1")
                return 1 if ((res.returncode == 0) == should_be_valid) else 0
            except Exception as ex:
                print(f"[validator candidate {i}] test error on {in_path}: {ex}")
                return 0
        with ThreadPoolExecutor() as ex:
            futures = [ex.submit(run_one, tp) for tp in val_tests]
            for fut in as_completed(futures):
                score += fut.result()
        print(f"[score] validator candidate {i}: {score}/{len(val_tests)}", flush=True)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx == -1:
        raise RuntimeError("All validator candidates failed to compile or scored 0")
    return best_idx, candidates[best_idx]


def _load_standard_checker_if_named(text: str) -> Optional[str]:
    name = text.strip()
    std_names = {"fcmp.cpp", "lcmp.cpp", "hcmp.cpp", "wcmp.cpp", "ncmp.cpp", "rcmp4.cpp", "rcmp6.cpp", "rcmp9.cpp", "yesno.cpp"}
    if name in std_names:
        std_path = os.path.join(STD_CHECKERS_DIR, name)
        if os.path.exists(std_path):
            return open(std_path, "r", encoding="utf-8").read()
    return None


def _find_standard_checker_in_text(text: str) -> Optional[str]:
    std_names = ["fcmp.cpp", "lcmp.cpp", "hcmp.cpp", "wcmp.cpp", "ncmp.cpp", "rcmp4.cpp", "rcmp6.cpp", "rcmp9.cpp", "yesno.cpp"]
    lowered = text.lower()
    for name in std_names:
        if name.lower() in lowered:
            std_path = os.path.join(STD_CHECKERS_DIR, name)
            if os.path.exists(std_path):
                return open(std_path, "r", encoding="utf-8").read()
    return None


def _score_checkers(pdir: str, candidates: list[str], chk_tests: list[tuple[str, str, str, str]]) -> tuple[int, str]:
    print(f"[score] checkers: {len(candidates)} candidates, {len(chk_tests)} tests", flush=True)
    best_idx = -1
    best_score = -1
    best_code: str = ""
    for i, content in enumerate(candidates):
        # Always store raw model output for diagnosis
        raw_path = os.path.join(pdir, f"checker_candidate_{i}.raw.txt")
        _write_file(raw_path, content)
        # Prefer standard checker mention anywhere in the text
        code = _find_standard_checker_in_text(content)
        if not code:
            code = _extract_cpp(content)
        if not code:
            maybe_std = _load_standard_checker_if_named(content)
            if not maybe_std:
                print(f"[checker candidate {i}] not C++ and not a known standard checker name: {content[:120]}")
                continue
            code = maybe_std
        src = os.path.join(pdir, f"checker_candidate_{i}.cpp")
        _write_file(src, code)
        print(f"[checker candidate {i}] source saved to {src}")
        print(f"[checker candidate {i}] code:\n{code}")
        try:
            bin_path = os.path.join(pdir, f"checker_candidate_{i}.out")
            _compile_cpp(src, bin_path)
        except Exception as e:
            print(f"[checker candidate {i}] compile error: {e}")
            continue
        score = 0
        # parallel per-test runs
        def run_one(test_quad: tuple[str, str, str, str]) -> int:
            in_path, out_path, ans_path, res_path = test_quad
            try:
                res = subprocess.run([bin_path, in_path, out_path, ans_path], capture_output=True, text=True, timeout=10)
                expected = open(res_path, "r", encoding="utf-8").read().strip()
                should_pass = (expected == "1")
                if (res.returncode == 0) == should_pass:
                    return 1
                else:
                    print(f"[checker candidate {i}] mismatch on {in_path} expected {expected} got rc={res.returncode}")
                    if res.stderr:
                        print(f"[checker candidate {i}] stderr: {res.stderr[:200]}")
                    return 0
            except Exception as ex:
                print(f"[checker candidate {i}] test error on {in_path}: {ex}")
                return 0
        with ThreadPoolExecutor() as ex:
            futures = [ex.submit(run_one, tq) for tq in chk_tests]
            for fut in as_completed(futures):
                score += fut.result()
        print(f"[score] checker candidate {i}: {score}/{len(chk_tests)}", flush=True)
        if score > best_score:
            best_score = score
            best_idx = i
            best_code = code
    if best_idx == -1:
        raise RuntimeError("All checker candidates failed to compile or scored 0")
    return best_idx, best_code


def _compile_cpp(src: str, out: str, include_testlib: bool = True) -> None:
    cmd = ["g++", "-std=c++17", "-O2", "-pipe", src, "-o", out]
    if include_testlib and os.path.exists(TESTLIB_PATH):
        # Copy testlib.h next to src so includes like "testlib.h" work
        try:
            shutil.copy(TESTLIB_PATH, os.path.join(os.path.dirname(src), "testlib.h"))
        except Exception:
            pass
    subprocess.run(cmd, check=True)


def _write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


async def _upload_problem_zip(pid: str, zip_path: str) -> None:
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("pid", pid)
        data.add_field("zipfile", open(zip_path, "rb"), filename=os.path.basename(zip_path), content_type="application/zip")
        async with session.post(f"{JUDGE_BASE_URL}/problem/add-problem", data=data) as r:
            if r.status != 200:
                txt = await r.text()
                if r.status == 500 and "already exists" in txt:
                    # Try setup on existing problem
                    data2 = aiohttp.FormData()
                    data2.add_field("pid", pid)
                    data2.add_field("zipfile", open(zip_path, "rb"), filename=os.path.basename(zip_path), content_type="application/zip")
                    async with session.post(f"{JUDGE_BASE_URL}/problem/setup", data=data2) as r2:
                        if r2.status != 200:
                            txt2 = await r2.text()
                            raise RuntimeError(f"Setup failed {r2.status}: {txt2}")
                        return
                raise RuntimeError(f"Upload failed {r.status}: {txt}")


def _zip_dir(dir_path: str, out_zip: str) -> None:
    # Ensure .zip extension
    if not out_zip.endswith(".zip"):
        out_zip = out_zip + ".zip"
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, dir_path)
                zf.write(abs_path, arcname=rel_path)


def _load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_prompts(statement: str, solution: str) -> tuple[str, str, str]:
    gen_tmpl = _load_template(GEN_TEMPLATE_PATH)
    val_tmpl = _load_template(VAL_TEMPLATE_PATH)
    chk_tmpl = _load_template(CHK_TEMPLATE_PATH)
    gen_prompt = gen_tmpl.replace("{{problem statement}}", statement)
    val_prompt = val_tmpl.replace("{{problem statement}}", statement).replace("{{accepted solution}}", solution)
    chk_prompt = chk_tmpl.replace("{{problem statement}}", statement).replace("{{accepted solution}}", solution)
    return gen_prompt, val_prompt, chk_prompt


def _write_config_yaml(problem_dir: str, time_limit: str = "2s", memory_limit: str = "512m") -> None:
    config = {
        "type": "default",
        "time_limit": time_limit,
        "memory_limit": memory_limit,
        "checker": "chk.cc",
        "input_prefix": "",
        "output_prefix": "",
        "input_suffix": ".in",
        "output_suffix": ".ans",
        "subtasks": [
            {"score": 100, "n_cases": NUM_TESTCASES}
        ]
    }
    _write_file(os.path.join(problem_dir, "config.yaml"), json.dumps(config, indent=2))


def build_tests_and_upload() -> None:
    if not os.path.exists(MANIFEST_PATH):
        raise RuntimeError(f"Manifest not found: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    ids = manifest.get("generated_ids", [])
    if not ids:
        print("No generated ids in manifest.")
        return

    sampling_client = _get_latest_sampler_client()
    renderer = _build_renderer()

    for pid in ids:
        pdir = os.path.join(OUTPUT_DIR, pid)
        print(f"[pipeline] processing {pid}", flush=True)
        stmt_path = os.path.join(pdir, "statement.txt")
        sol_path = os.path.join(pdir, "solution.cpp")
        if not os.path.exists(stmt_path) or not os.path.exists(sol_path):
            print(f"Skipping {pid}: missing statement or solution")
            continue
        statement = open(stmt_path, "r", encoding="utf-8").read()
        solution = open(sol_path, "r", encoding="utf-8").read()
        # If solution contains markdown fences, strip them
        sol_cpp_only = _extract_cpp(solution)
        if sol_cpp_only:
            with open(sol_path, "w", encoding="utf-8") as f:
                f.write(sol_cpp_only)
            solution = sol_cpp_only

        # Generate test suites via Tinker (skip problem on failure)
        try:
            val_tests = _generate_validator_tests(pdir, statement, sampling_client)
        except Exception as e:
            print(f"[skip] validator tests generation failed for {pid}: {e}")
            continue
        try:
            chk_tests = _generate_checker_tests(pdir, statement, solution, sampling_client)
        except Exception as e:
            print(f"[skip] checker tests generation failed for {pid}: {e}")
            continue

        attempt_ok = False
        for attempt in range(3):
            t_attempt = time.time()
            # Clean any previous artifacts
            for fname in ("generator.cpp", "validator.cpp", "chk.cc", "generator.out", "validator.out", "solution.out"):
                fpath = os.path.join(pdir, fname)
                try:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                except Exception:
                    pass
            testdata_dir = os.path.join(pdir, "testdata")
            if os.path.exists(testdata_dir):
                shutil.rmtree(testdata_dir, ignore_errors=True)

            # Generate candidates (K=3) for validator/checker/generator
            gen_prompt, val_prompt, chk_prompt = _build_prompts(statement, solution)
            try:
                gen_codes = _generate_candidates_from_template(GEN_TEMPLATE_PATH, statement, None, "generator", sampling_client, NUM_GENERATOR_CANDIDATES)
                val_codes = _generate_candidates_from_template(VAL_TEMPLATE_PATH, statement, solution, "validator", sampling_client, NUM_VALIDATOR_CANDIDATES)
                chk_codes = _generate_candidates_from_template(CHK_TEMPLATE_PATH, statement, solution, "checker", sampling_client, NUM_CHECKER_CANDIDATES)
            except Exception as e:
                print(f"[attempt {attempt+1}] candidate generation failed: {e}")
                continue

            # Select winner validator and checker by scoring on generated tests
            try:
                best_val_idx, best_val_code = _score_validators(pdir, val_codes, val_tests)
                print(f"[select] validator winner idx={best_val_idx}", flush=True)
            except Exception as e:
                print(f"Validator selection failed: {e}")
                continue
            try:
                best_chk_idx, best_chk_code = _score_checkers(pdir, chk_codes, chk_tests)
                print(f"[select] checker winner idx={best_chk_idx}", flush=True)
            except Exception as e:
                print(f"Checker selection failed: {e}")
                continue

            # Write sources
            gen_src = os.path.join(pdir, "generator.cpp")
            val_src = os.path.join(pdir, "validator.cpp")
            chk_src = os.path.join(pdir, "chk.cc")
            # Choose the generator that yields most valid cases below; initialize with first, will recompile others in selection
            # For now write first; we'll select below
            _write_file(val_src, best_val_code)
            _write_file(chk_src, best_chk_code)

            # Compile sources
            gen_bin = os.path.join(pdir, "generator.out")
            val_bin = os.path.join(pdir, "validator.out")
            sol_bin = os.path.join(pdir, "solution.out")
            try:
                _compile_cpp(val_src, val_bin)
                _compile_cpp(sol_path, sol_bin, include_testlib=False)
            except Exception as e:
                print(f"[attempt {attempt+1}] compile failure (validator/solution): {e}")
                continue

            # Select best generator among candidates by #valid produced
            best_gen_idx = -1
            best_valid = -1
            for i, gen_code in enumerate(gen_codes):
                _write_file(gen_src, gen_code)
                try:
                    _compile_cpp(gen_src, gen_bin)
                except Exception:
                    continue
                # quick probe of small number (e.g., 5) to rank
                valid_tmp = 0
                for j in range(1, min(6, NUM_TESTCASES) + 1):
                    try:
                        proc = subprocess.run([gen_bin, str(j)], capture_output=True, text=True, timeout=10)
                        if proc.returncode != 0:
                            continue
                        tmp_in = os.path.join(pdir, f"tmp_{j}.in")
                        with open(tmp_in, "w", encoding="utf-8") as f:
                            f.write(proc.stdout)
                        with open(tmp_in, "r", encoding="utf-8") as fin:
                            vres = subprocess.run([val_bin], stdin=fin, capture_output=True, text=True, timeout=10)
                        os.remove(tmp_in)
                        if vres.returncode == 0:
                            valid_tmp += 1
                    except Exception:
                        pass
                if valid_tmp > best_valid:
                    best_valid = valid_tmp
                    best_gen_idx = i
            if best_gen_idx == -1:
                print("All generator candidates failed; retrying")
                continue

            # Use best generator for final dataset
            _write_file(gen_src, gen_codes[best_gen_idx])
            _compile_cpp(gen_src, gen_bin)

            # Generate test cases and validate with winner validator
            os.makedirs(testdata_dir, exist_ok=True)
            valid_count = 0
            print(f"[final] generating {NUM_TESTCASES} final testcases", flush=True)
            t_gen = time.time()
            for i in range(1, NUM_TESTCASES + 1):
                try:
                    gen = subprocess.run([gen_bin, str(i)], capture_output=True, text=True, timeout=10)
                    if gen.returncode != 0:
                        continue
                    case_in = os.path.join(testdata_dir, f"{i}.in")
                    with open(case_in, "w", encoding="utf-8") as f:
                        f.write(gen.stdout)

                    # Validate
                    with open(case_in, "r", encoding="utf-8") as fin:
                        vres = subprocess.run([val_bin], stdin=fin, capture_output=True, text=True, timeout=10)
                    if vres.returncode != 0:
                        os.remove(case_in)
                        continue

                    # Produce .ans using solution
                    with open(case_in, "r", encoding="utf-8") as fin2:
                        sres = subprocess.run([sol_bin], stdin=fin2, capture_output=True, text=True, timeout=10)
                    if sres.returncode != 0:
                        os.remove(case_in)
                        continue
                    case_ans = os.path.join(testdata_dir, f"{i}.ans")
                    with open(case_ans, "w", encoding="utf-8") as fa:
                        fa.write(sres.stdout)
                    valid_count += 1
                except Exception:
                    continue

            print(f"[final] generated {valid_count}/{NUM_TESTCASES} valid tests in {time.time()-t_gen:.2f}s", flush=True)
            if valid_count > 0:
                attempt_ok = True
                break
            else:
                print(f"Attempt {attempt+1}/3: no valid test cases for {pid}; retrying generation")

        if not attempt_ok:
            print(f"No valid test cases for {pid}; skipping upload")
            continue
        print(f"[attempt] completed in {time.time()-t_attempt:.2f}s", flush=True)

        # Write config.yaml
        _write_config_yaml(pdir)

        # Create tar.gz (API expects zipfile; server accepts any filename)
        zip_path = os.path.join(OUTPUT_DIR, f"{pid}.zip")
        _zip_dir(pdir, zip_path)

        # Upload (skip problem if upload fails)
        try:
            asyncio.run(_upload_problem_zip(pid, zip_path))
            print(f"Uploaded problem {pid} to judge")
        except Exception as e:
            print(f"[skip] upload failed for {pid}: {e}")
            continue

    # Update RL dataset jsonl for generated problems
    ds_path = os.path.join(OUTPUT_DIR, "dataset.jsonl")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    ids = manifest.get("generated_ids", [])
    mode = "w" if os.environ.get("SELF_IMPROVE_RESET_DATASET", "") == "1" else "a"
    with open(ds_path, mode, encoding="utf-8") as f:
        for pid in ids:
            stmt_file = os.path.join(OUTPUT_DIR, pid, "statement.txt")
            if os.path.exists(stmt_file):
                stmt = open(stmt_file, "r", encoding="utf-8").read().strip()
                f.write(json.dumps({"problem_id": pid, "problem_statement": stmt}) + "\n")


if __name__ == "__main__":
    build_tests_and_upload()


