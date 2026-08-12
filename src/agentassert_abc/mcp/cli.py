# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""``agentassert-abc-mcp-guard`` — wrap an MCP server in a behavioral contract.

Uses :mod:`argparse` rather than click on purpose. This is the universal
adoption surface, so it must install with nothing heavier than
``agentassert-abc[gateway]`` — dragging in the proxy's web stack to launch a
stdio relay would be a poor trade.
"""

from __future__ import annotations

import argparse
import sys

from agentassert_abc.exceptions import ContractLoadError
from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.gateway.state import (
    MCP_PROVIDED_FIELDS,
    assert_evaluable_on_response_surface,
)
from agentassert_abc.mcp.guard import McpGuard
from agentassert_abc.mcp.interposer import run_guard

__all__ = ["build_parser", "main"]

_EPILOG = """\
examples:
  agentassert-abc-mcp-guard --contract c.yaml -- npx -y @modelcontextprotocol/server-github
  agentassert-abc-mcp-guard --contract c.yaml --fail-closed -- python -m my_server

Put the same line in your client's MCP config as the server "command"/"args"
to enforce the contract in Claude Code, Codex, Cursor, VS Code or Antigravity.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentassert-abc-mcp-guard",
        description="Enforce an AgentAssert behavioral contract on MCP tool calls.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--contract", required=True, help="path to the contract YAML")
    parser.add_argument(
        "--server-label",
        default="mcp",
        help="name recorded as `tool.server` (default: mcp)",
    )
    parser.add_argument("--session-id", default=None, help="stable session id")
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help="deny a tool call the guard cannot evaluate (default: allow it through)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="`--` followed by the downstream MCP server's command",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print(
            "[agentassert] no downstream server command given; "
            "pass it after `--`, e.g. `-- npx -y some-server`",
            file=sys.stderr,
        )
        return 2

    try:
        enforcer = SessionEnforcer.from_yaml(args.contract)
        # Refuse a contract this surface can never evaluate, rather than scoring
        # the agent as violating it on every call. Unlike the in-process hooks,
        # this is a hard failure: the guard is launched by the client at startup,
        # so a loud exit is visible in the client's MCP server log, where a
        # silently-disabled guard would not be.
        assert_evaluable_on_response_surface(enforcer._contract, "MCP guard", MCP_PROVIDED_FIELDS)
    except ContractLoadError as exc:
        print(f"[agentassert] contract not loaded: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"[agentassert] could not start guard: {exc}", file=sys.stderr)
        return 1

    guard = McpGuard(
        enforcer,
        server_label=args.server_label,
        session_id=args.session_id,
        fail_closed=args.fail_closed,
    )
    print(
        f"[agentassert] guarding '{args.server_label}' with contract "
        f"'{getattr(enforcer._contract, 'name', args.contract)}' "
        f"(session {guard.session_id}, "
        f"{'fail-closed' if args.fail_closed else 'fail-open'})",
        file=sys.stderr,
    )
    try:
        return run_guard(guard, command)
    except OSError as exc:
        # A missing or non-executable downstream binary is the most common
        # misconfiguration. Report it on stderr and exit: letting the traceback
        # escape would print a non-JSON blob into the stream the client is
        # parsing as JSON-RPC.
        print(
            f"[agentassert] could not start downstream server {command[0]!r}: {exc}",
            file=sys.stderr,
        )
        return 1


def cli() -> None:
    """Console-script entry point."""
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
