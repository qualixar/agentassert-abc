# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""`agentassert-abc-proxy` CLI — start/inspect the enforcement proxy.

Ported from agentassert-typec's `cli.py` (entry point renamed from
`agentassert-proxy` to `agentassert-abc-proxy`; repo/env-var references
updated to this package's identity).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import uvicorn

_ANTHROPIC_DEFAULT = "https://api.anthropic.com"
_OPENAI_DEFAULT = "https://api.openai.com"


def _warn_if_upstream_mismatch(contract_path: Path) -> None:
    """Warn if env vars point to a non-default backend but the contract has
    no upstream override.
    """
    from ruamel.yaml import YAML

    try:
        data = YAML().load(contract_path.read_text())
    except Exception:
        return

    has_upstream = isinstance(data, dict) and bool(data.get("upstream"))
    anthropic_env = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    openai_env = os.environ.get("OPENAI_BASE_URL", "").strip()

    warnings = []
    if anthropic_env and not anthropic_env.startswith(_ANTHROPIC_DEFAULT) and not has_upstream:
        warnings.append(f"  ANTHROPIC_BASE_URL={anthropic_env}")
    if openai_env and not openai_env.startswith(_OPENAI_DEFAULT) and not has_upstream:
        warnings.append(f"  OPENAI_BASE_URL={openai_env}")

    if warnings:
        click.echo("Warning: non-default upstream detected but contract has no 'upstream:' block.")
        click.echo("The proxy will forward to these env-var URLs automatically, but you can")
        click.echo("also pin them explicitly in your contract for reproducibility:")
        click.echo()
        click.echo("  upstream:")
        for w in warnings:
            key = "anthropic" if "ANTHROPIC" in w else "openai"
            url = w.split("=", 1)[1]
            click.echo(f"    {key}: {url}")
        click.echo()


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--contract", "-c", required=True, help="Path to contract YAML")
@click.option("--port", "-p", default=9000, help="Port to listen on")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option(
    "--session-id",
    "-s",
    default=None,
    help="Session ID for DB isolation (creates separate .db file per ID)",
)
@click.option(
    "--no-persist",
    is_flag=True,
    default=False,
    help="Disable SQLite session persistence (in-memory only)",
)
def start(contract: str, port: int, host: str, session_id: str | None, no_persist: bool) -> None:
    contract_path = Path(contract).resolve()
    if not contract_path.exists():
        click.echo(f"Error: Contract file not found: {contract_path}", err=True)
        sys.exit(1)

    from agentassert_abc._version import __version__

    click.echo(f"AgentAssert ABC Proxy v{__version__}")
    click.echo(f"Contract: {contract_path}")
    click.echo(f"Listening on http://{host}:{port}")
    if no_persist:
        click.echo("Persistence: disabled (--no-persist)")
    else:
        click.echo(f"Persistence: enabled (session_id={session_id or 'default'})")
    click.echo()
    _warn_if_upstream_mismatch(contract_path)
    click.echo("Set these env vars in your agent environment:")
    click.echo(f"  export ANTHROPIC_BASE_URL=http://{host}:{port}/anthropic")
    click.echo(f"  export OPENAI_BASE_URL=http://{host}:{port}/openai")
    click.echo()

    from agentassert_abc.proxy.server import create_app

    app = create_app(
        contract_path=str(contract_path),
        session_id=session_id,
        persist=not no_persist,
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


@cli.command()
@click.option("--port", "-p", default=9000, help="Port of running proxy")
@click.option("--host", "-h", default="127.0.0.1", help="Host of running proxy")
def status(port: int, host: str) -> None:
    import httpx

    try:
        resp = httpx.get(f"http://{host}:{port}/health", timeout=2.0)
        click.echo(resp.text)
    except httpx.HTTPError as e:
        click.echo(f"Proxy not reachable: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
