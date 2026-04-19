"""
AutoCode grading — code extraction and remote judge client.

This module handles two concerns:

1. **Code extraction**: parse the model's response for the last fenced code
   block and detect its language (C++ or Python).
2. **Remote judge**: submit the extracted code to a judge server via HTTP
   and poll for the pass/fail verdict.

The judge address is configured via environment variables:

- ``JUDGE_HOST`` (default: ``localhost``)
- ``JUDGE_PORT`` (default: ``8081``)

Judge API contract:

- ``POST /submit`` with ``{"pid", "lang", "code"}`` → ``{"sid": int}``
- ``GET /result/{sid}`` → ``{"status": "done"|"pending", "passed": bool}``
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

import aiohttp

logger = logging.getLogger(__name__)


def get_judge_base_url() -> str:
    host = os.getenv("JUDGE_HOST", "localhost")
    port = os.getenv("JUDGE_PORT", "8081")
    return f"http://{host}:{port}"


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def extract_code_from_response(response: str) -> tuple[str, str] | None:
    """Extract the last fenced code block and detect its language.

    Returns:
        (code, lang) where lang is "cpp" or "py", or None if no block found.
    """
    matches = list(_CODE_BLOCK_RE.finditer(response))
    if not matches:
        return None

    last = matches[-1]
    lang_tag = last.group(1).strip().lower()
    code = last.group(2).strip()

    lang = _normalize_language(lang_tag, code)
    return code, lang


def _normalize_language(lang_tag: str, code: str) -> str:
    if lang_tag in ("cpp", "c++", "cxx", "c"):
        return "cpp"
    if lang_tag in ("python", "py", "python3"):
        return "py"
    if lang_tag == "":
        return _detect_language(code)
    # Unknown tag — try detection as fallback
    return _detect_language(code)


_CPP_INDICATORS = (
    "#include", "using namespace std", "std::",
    "cout", "cin", "endl", "vector<", "#define", "int main",
)


def _detect_language(code: str) -> str:
    code_lower = code.lower()
    if any(indicator in code_lower for indicator in _CPP_INDICATORS):
        return "cpp"
    return "py"


# ---------------------------------------------------------------------------
# Remote judge interaction
# ---------------------------------------------------------------------------


async def judge_code(code: str, lang: str, problem_id: str) -> tuple[bool, dict]:
    """Submit code to remote judge and wait for result.

    Returns:
        (passed, details_dict)
    """
    base_url = get_judge_base_url()

    sid = await _submit(base_url, problem_id, lang, code)
    if sid is None:
        return False, {"error": "submit failed"}

    result = await _poll_result(base_url, sid)
    if result is None:
        return False, {"error": "timeout waiting for judge result"}

    passed = result.get("passed", False)
    return passed, result


async def _submit(base_url: str, pid: str, lang: str, code: str) -> int | None:
    payload = {"pid": pid, "lang": lang, "code": code}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/submit", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("sid")
                logger.warning("Judge submit returned status %d", resp.status)
    except Exception as e:
        logger.warning("Judge submit failed: %s", e)
    return None


async def _poll_result(
    base_url: str,
    sid: int,
    timeout: int = 60,
    interval: float = 1.0,
) -> dict | None:
    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(int(timeout / interval)):
                async with session.get(f"{base_url}/result/{sid}") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "done":
                            return data
                await asyncio.sleep(interval)
    except Exception as e:
        logger.warning("Judge poll failed: %s", e)
    return None
