from __future__ import annotations

import asyncio

import chz

from tinker_cookbook.autocode_rl.eval.run import CLIConfig, cli_main


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(chz.replace(cli_config, benchmark="gpqa")))

