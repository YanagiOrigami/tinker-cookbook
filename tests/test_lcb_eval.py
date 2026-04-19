"""Quick diagnostic for LCB eval pipeline.

Run with:
    python tests/test_lcb_eval.py
"""

import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s:%(lineno)d [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main():
    # --- Step 1: Load a few LCB tasks ---
    logger.info("=== Step 1: Loading LCB tasks ===")
    from tinker_cookbook.recipes.autocode_rl.autocode_env import load_lcb_tasks

    tasks = load_lcb_tasks("test")
    logger.info("Loaded %d tasks", len(tasks))
    if not tasks:
        logger.error("No tasks loaded! Check HuggingFace connectivity.")
        return

    task = tasks[0]
    logger.info("First task tests (%d cases):", len(task.tests))
    for i, t in enumerate(task.tests[:2]):
        logger.info("  test[%d]: testtype=%s, input=%s..., output=%s...",
                     i, t.get("testtype"), t.get("input", "")[:80], t.get("output", "")[:80])

    # --- Step 2: Check sandbox connectivity ---
    logger.info("\n=== Step 2: Checking sandbox connectivity ===")
    import aiohttp
    import os
    sandbox_url = os.getenv("SANDBOX_URL", "http://localhost:8080/run_code")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                sandbox_url,
                json={"code": "print('hello')", "files": {}, "timeout": 5},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                logger.info("Sandbox responded: status=%d", resp.status)
                body = await resp.text()
                logger.info("Sandbox body: %s", body[:200])
    except Exception as e:
        logger.error("Sandbox NOT reachable at %s: %s", sandbox_url, e)
        logger.error("Start it with: docker run -d -p 8080:8080 volcengine/sandbox-fusion:server-20250609")
        logger.info("Skipping remaining tests that require sandbox.")
        return

    # --- Step 3: Test code extraction ---
    logger.info("\n=== Step 3: Testing code extraction ===")
    from tinker_cookbook.recipes.code_rl.code_grading import extract_code_from_model

    fake_response = '```python\nprint("hello world")\n```'
    code = extract_code_from_model(fake_response)
    logger.info("Extracted code: %r", code)
    if code is None:
        logger.error("Code extraction failed!")
        return

    # --- Step 4: Test sandbox_check_correctness with a known-good solution ---
    logger.info("\n=== Step 4: Testing sandbox_check_correctness ===")
    from tinker_cookbook.recipes.code_rl.code_grading import sandbox_check_correctness

    # Simple stdin/stdout test
    simple_tests = [
        {"input": "3\n", "output": "3\n", "testtype": "stdin_stdout", "metadata": {}},
    ]
    simple_code = "x = input()\nprint(x)"

    try:
        passed, details = await sandbox_check_correctness(simple_tests, simple_code, timeout=6)
        logger.info("Simple test: passed=%s, details=%s", passed, details)
    except Exception as e:
        logger.error("sandbox_check_correctness failed: %s", e)
        return

    # --- Step 5: Test with an actual LCB task ---
    logger.info("\n=== Step 5: Testing with actual LCB task (expecting fail with dummy code) ===")
    dummy_code = "pass"
    try:
        passed, details = await sandbox_check_correctness(task.tests, dummy_code, timeout=6)
        logger.info("LCB task with dummy code: passed=%s (expected False)", passed)
    except Exception as e:
        logger.error("sandbox_check_correctness with LCB task failed: %s", e)

    # --- Step 6: Check a functional test if any ---
    logger.info("\n=== Step 6: Looking for a functional test task ===")
    for t in tasks[:20]:
        if t.tests and t.tests[0].get("testtype") == "functional":
            logger.info("Found functional task, func_name=%s", t.tests[0].get("metadata", {}).get("func_name"))
            try:
                passed, details = await sandbox_check_correctness(t.tests, "def f(): return 0", timeout=6)
                logger.info("Functional test with dummy: passed=%s", passed)
            except Exception as e:
                logger.error("Functional test failed: %s", e)
            break
    else:
        logger.info("No functional test tasks found in first 20 tasks")

    logger.info("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
