# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""The stdio pump that sits between an MCP client and an MCP server.

All I/O and no policy — :mod:`agentassert_abc.mcp.guard` owns every decision.
:class:`StreamPump` is written against plain text streams rather than a
subprocess so the whole relay can be exercised with in-memory pipes; the
subprocess is wired up only in :func:`run_guard`.

Deployment is a config edit in the MCP client, with no vendor-specific code::

    {"command": "npx", "args": ["-y", "some-server"]}

becomes::

    {"command": "agentassert-abc-mcp-guard",
     "args": ["--contract", "contract.yaml", "--", "npx", "-y", "some-server"]}

which is why one artifact covers Claude Code, Codex, Cursor, VS Code,
Antigravity, Windsurf and anything else that speaks MCP.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import TYPE_CHECKING, TextIO

from agentassert_abc.mcp import jsonrpc

if TYPE_CHECKING:
    from agentassert_abc.mcp.guard import McpGuard

__all__ = ["StreamPump", "run_guard"]


class StreamPump:
    """Relays newline-delimited JSON-RPC between a client and a server.

    Args:
        guard: the policy to apply.
        client_in: where the MCP client's requests arrive (this process' stdin).
        client_out: where replies to the client go (this process' stdout).
        server_in: the downstream server's stdin.
        server_out: the downstream server's stdout.
    """

    def __init__(
        self,
        guard: McpGuard,
        *,
        client_in: TextIO,
        client_out: TextIO,
        server_in: TextIO,
        server_out: TextIO,
    ) -> None:
        self._guard = guard
        self._client_in = client_in
        self._client_out = client_out
        self._server_in = server_in
        self._server_out = server_out
        # Both directions can reach client_out: the server relay writes
        # responses, and the client relay writes synthesised DENY replies.
        # Interleaving them would split a line and corrupt the framing.
        self._out_lock = threading.Lock()

    # -- directions ---------------------------------------------------------

    def pump_client_to_server(self) -> None:
        """Read client requests, screen them, forward or answer directly."""
        for line in self._client_in:
            message = jsonrpc.decode(line)
            if message is None:
                self._write(self._server_in, line if line.endswith("\n") else line + "\n")
                continue

            relay = self._guard.on_client_message(message)
            if relay.reply is not None:
                # A DENY: answer the client without the server ever seeing it.
                self._write_client(jsonrpc.encode(relay.reply))
            if relay.forward is not None:
                self._write(self._server_in, jsonrpc.encode(relay.forward))
        self._close(self._server_in)

    def pump_server_to_client(self) -> None:
        """Read server responses, score them, forward possibly rewritten."""
        for line in self._server_out:
            message = jsonrpc.decode(line)
            if message is None:
                self._write_client(line if line.endswith("\n") else line + "\n")
                continue

            # Only `forward` is meaningful here: a response travelling back to
            # the client has no second peer to short-circuit to.
            relay = self._guard.on_server_message(message)
            if relay.forward is not None:
                self._write_client(jsonrpc.encode(relay.forward))

    def run(self) -> None:
        """Run both directions until the client closes stdin and the server drains."""
        server_reader = threading.Thread(
            target=self._guarded(self.pump_server_to_client),
            name="agentassert-mcp-server-reader",
            daemon=True,
        )
        server_reader.start()
        self._guarded(self.pump_client_to_server)()
        server_reader.join(timeout=_DRAIN_TIMEOUT_S)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _guarded(fn):  # type: ignore[no-untyped-def]
        """Swallow the pipe errors that are normal at shutdown.

        When either end goes away mid-relay the OS raises rather than returning
        EOF. That is an ordinary end-of-session, not a fault, and must not print
        a traceback into a stream the client is still parsing as JSON-RPC.
        """

        def _run() -> None:
            try:
                fn()
            except (BrokenPipeError, ValueError, OSError):
                return

        return _run

    def _write_client(self, text: str) -> None:
        with self._out_lock:
            self._write(self._client_out, text)

    @staticmethod
    def _write(stream: TextIO, text: str) -> None:
        stream.write(text)
        stream.flush()

    @staticmethod
    def _close(stream: TextIO) -> None:
        try:
            stream.close()
        except (BrokenPipeError, OSError):
            return


_DRAIN_TIMEOUT_S = 5.0


def run_guard(
    guard: McpGuard,
    command: list[str],
    *,
    client_in: TextIO | None = None,
    client_out: TextIO | None = None,
) -> int:
    """Spawn the downstream MCP server and relay through ``guard``.

    Args:
        guard: policy to apply to ``tools/call`` traffic.
        command: the downstream server's argv, e.g. ``["npx", "-y", "srv"]``.
        client_in: defaults to this process' stdin.
        client_out: defaults to this process' stdout.

    Returns:
        The downstream server's exit code.
    """
    proc = subprocess.Popen(  # noqa: S603 — argv comes from the user's own MCP config.
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # downstream diagnostics stay on this process' stderr
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    if proc.stdin is None or proc.stdout is None:  # pragma: no cover - Popen contract
        msg = "failed to open pipes to the downstream MCP server"
        raise RuntimeError(msg)

    pump = StreamPump(
        guard,
        client_in=client_in if client_in is not None else sys.stdin,
        client_out=client_out if client_out is not None else sys.stdout,
        server_in=proc.stdin,
        server_out=proc.stdout,
    )
    try:
        pump.run()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_DRAIN_TIMEOUT_S)
            except subprocess.TimeoutExpired:  # pragma: no cover - slow child
                proc.kill()
    return proc.returncode or 0
