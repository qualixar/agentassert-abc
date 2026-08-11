# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Install/uninstall the Claude Code hook script.

Ported from agentassert-typec-claude-code's `installer.py`. Hook filename
renamed from `00-agentassert-typec.py` to `00-agentassert-abc.py` to match
this package's identity; repository URL updated accordingly.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from shutil import copy2

_HOOK_FILENAME = "00-agentassert-abc.py"


def install(contract_path: str, force: bool = False) -> None:
    hooks_dir = Path.home() / ".claude-hooks"
    hooks_dir.mkdir(exist_ok=True)

    hook_file = hooks_dir / _HOOK_FILENAME
    source_hook = Path(__file__).parent / "hook.py"

    if hook_file.exists() and not force:
        raise FileExistsError(f"Hook already installed at {hook_file}. Use --force to overwrite.")

    copy2(source_hook, hook_file)
    hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC | stat.S_IXUSR | stat.S_IXGRP)

    print(f"AgentAssert hook installed to {hook_file}")
    print()
    print("Add this to your shell config (~/.zshrc or ~/.bashrc):")
    print(f"  export AGENTASSERT_CONTRACT={contract_path}")
    print()
    if sys.stdout.isatty():
        print("If AgentAssert is useful, please star the repo:")
        print("  https://github.com/qualixar/agentassert-abc")


def uninstall() -> None:
    hook_file = Path.home() / ".claude-hooks" / _HOOK_FILENAME
    if hook_file.exists():
        hook_file.unlink()
        print(f"AgentAssert hook removed from {hook_file}")
    else:
        print("Hook not installed.")
