# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""`agentassert-abc-claude-code` CLI — install/uninstall/status the hook.

Ported from agentassert-typec-claude-code's `cli.py`. Entry point renamed
from `agentassert-claude-code` to `agentassert-abc-claude-code`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

_HOOK_FILENAME = "00-agentassert-abc.py"


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--contract", "-c", required=True, help="Path to contract YAML")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing hook")
def install_cmd(contract: str, force: bool) -> None:
    from agentassert_abc.claude_code.installer import install

    contract_path = Path(contract).resolve()
    if not contract_path.exists():
        click.echo(f"Error: Contract file not found: {contract_path}", err=True)
        sys.exit(1)

    install(str(contract_path), force=force)


@cli.command()
def uninstall_cmd() -> None:
    from agentassert_abc.claude_code.installer import uninstall

    uninstall()


@cli.command()
def status_cmd() -> None:
    hook_file = Path.home() / ".claude-hooks" / _HOOK_FILENAME
    if hook_file.exists():
        click.echo(f"Hook installed: {hook_file}")
    else:
        click.echo("Hook not installed")

    contract = os.environ.get("AGENTASSERT_CONTRACT", "")
    if contract:
        click.echo(f"   AGENTASSERT_CONTRACT={contract}")
    else:
        click.echo("   AGENTASSERT_CONTRACT not set")


if __name__ == "__main__":
    cli()
