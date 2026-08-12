# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""The stdio pump, in memory and against a real subprocess."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path  # noqa: TC003 — runtime use in a pytest fixture signature

import pytest

from agentassert_abc.mcp import jsonrpc
from agentassert_abc.mcp.guard import McpGuard
from agentassert_abc.mcp.interposer import StreamPump, run_guard

from .conftest import CONTRACTS, StubEnforcer, allow, deny, tool_call

# ---------------------------------------------------------------------------
# In-memory plumbing
# ---------------------------------------------------------------------------


class KeptStringIO(io.StringIO):
    """StringIO whose contents survive ``close()``.

    The pump closes the server's stdin to signal EOF, and a plain StringIO
    discards its buffer on close — which would make every assertion below read
    an empty string and pass for the wrong reason.
    """

    def close(self) -> None:  # noqa: D102
        return


def _lines(stream: io.StringIO) -> list[dict]:
    return [
        msg for line in stream.getvalue().splitlines() if (msg := jsonrpc.decode(line)) is not None
    ]


def _pump(guard: McpGuard, client_lines: list[str]) -> tuple[KeptStringIO, KeptStringIO]:
    """Run the client->server direction over in-memory streams."""
    server_in, client_out = KeptStringIO(), KeptStringIO()
    pump = StreamPump(
        guard,
        client_in=io.StringIO("".join(client_lines)),
        client_out=client_out,
        server_in=server_in,
        server_out=io.StringIO(""),
    )
    pump.pump_client_to_server()
    return server_in, client_out


class TestClientToServer:
    def test_allowed_call_reaches_the_server(self) -> None:
        guard = McpGuard(StubEnforcer([allow()]))
        server_in, client_out = _pump(guard, [jsonrpc.encode(tool_call(1, "read"))])
        assert len(_lines(server_in)) == 1
        assert _lines(server_in)[0]["params"]["name"] == "read"
        assert client_out.getvalue() == ""

    def test_denied_call_never_reaches_the_server(self) -> None:
        guard = McpGuard(StubEnforcer([deny(reason="blocked")]))
        server_in, client_out = _pump(guard, [jsonrpc.encode(tool_call(1, "rm"))])
        assert server_in.getvalue() == "", "a denied call must not cross the boundary"
        replies = _lines(client_out)
        assert len(replies) == 1
        assert replies[0]["result"]["isError"] is True

    def test_unparseable_line_is_relayed_verbatim(self) -> None:
        # Not our protocol to police; dropping it would break the peer.
        guard = McpGuard(StubEnforcer())
        server_in, _ = _pump(guard, ["this is not json\n"])
        assert server_in.getvalue() == "this is not json\n"

    def test_final_line_without_a_newline_is_terminated(self) -> None:
        guard = McpGuard(StubEnforcer())
        server_in, _ = _pump(guard, ["not json"])
        assert server_in.getvalue().endswith("\n")

    def test_blank_lines_are_preserved_as_framing(self) -> None:
        guard = McpGuard(StubEnforcer())
        server_in, _ = _pump(guard, ["\n"])
        assert server_in.getvalue() == "\n"

    def test_mixed_traffic_keeps_order_and_blocks_only_the_denied_call(self) -> None:
        guard = McpGuard(StubEnforcer([allow(), deny(), allow()]))
        server_in, client_out = _pump(
            guard,
            [
                jsonrpc.encode({"jsonrpc": "2.0", "id": 0, "method": "initialize"}),
                jsonrpc.encode(tool_call(1, "ok-one")),
                jsonrpc.encode(tool_call(2, "denied")),
                jsonrpc.encode(tool_call(3, "ok-two")),
            ],
        )
        forwarded = [m.get("params", {}).get("name", m.get("method")) for m in _lines(server_in)]
        assert forwarded == ["initialize", "ok-one", "ok-two"]
        assert [m["id"] for m in _lines(client_out)] == [2]


class TestServerToClient:
    def test_response_is_relayed_to_the_client(self) -> None:
        guard = McpGuard(StubEnforcer())
        guard.on_client_message(tool_call(1, "read"))
        client_out = KeptStringIO()
        pump = StreamPump(
            guard,
            client_in=io.StringIO(""),
            client_out=client_out,
            server_in=KeptStringIO(),
            server_out=io.StringIO(
                jsonrpc.encode(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {"content": [{"type": "text", "text": "hi"}]},
                    }
                )
            ),
        )
        pump.pump_server_to_client()
        assert _lines(client_out)[0]["result"]["content"][0]["text"] == "hi"

    def test_server_stdout_noise_is_relayed_verbatim(self) -> None:
        guard = McpGuard(StubEnforcer())
        client_out = KeptStringIO()
        pump = StreamPump(
            guard,
            client_in=io.StringIO(""),
            client_out=client_out,
            server_in=KeptStringIO(),
            server_out=io.StringIO("startup banner\n"),
        )
        pump.pump_server_to_client()
        assert client_out.getvalue() == "startup banner\n"


class TestShutdown:
    def test_broken_pipe_during_relay_is_not_a_crash(self) -> None:
        # The peer going away mid-relay is an ordinary end of session. A
        # traceback here would land in a stream the client parses as JSON-RPC.
        class Exploding(io.StringIO):
            def write(self, _s: str) -> int:
                raise BrokenPipeError

        guard = McpGuard(StubEnforcer([allow()]))
        pump = StreamPump(
            guard,
            client_in=io.StringIO(jsonrpc.encode(tool_call(1, "read"))),
            client_out=KeptStringIO(),
            server_in=Exploding(),
            server_out=io.StringIO(""),
        )
        pump.run()  # must return, not raise


# ---------------------------------------------------------------------------
# Real subprocess — in-memory pipes do not prove the wiring
# ---------------------------------------------------------------------------

FAKE_SERVER = textwrap.dedent(
    """
    import json, sys
    seen = open(sys.argv[1], "w")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        seen.write(msg.get("params", {}).get("name", msg.get("method", "?")) + "\\n")
        seen.flush()
        if msg.get("id") is not None:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0", "id": msg["id"],
                "result": {"content": [{"type": "text", "text": "ran"}]},
            }) + "\\n")
            sys.stdout.flush()
    seen.close()
    """
)


@pytest.fixture
def fake_server(tmp_path: Path) -> tuple[list[str], Path]:
    script = tmp_path / "fake_mcp_server.py"
    script.write_text(FAKE_SERVER)
    seen = tmp_path / "seen.txt"
    return [sys.executable, str(script), str(seen)], seen


class TestRunGuardEndToEnd:
    def test_allowed_call_reaches_the_real_server_and_returns(
        self, fake_server, safety_enforcer
    ) -> None:
        command, seen = fake_server
        client_out = KeptStringIO()
        code = run_guard(
            McpGuard(safety_enforcer),
            command,
            client_in=io.StringIO(jsonrpc.encode(tool_call(1, "Read", {"path": "a"}))),
            client_out=client_out,
        )
        assert code == 0
        assert "Read" in seen.read_text()
        assert _lines(client_out)[0]["result"]["content"][0]["text"] == "ran"

    def test_denied_call_never_reaches_the_real_server(self, fake_server, safety_enforcer) -> None:
        # The claim the whole guard exists to make, proven across a real process
        # boundary: the downstream server records every message it receives, and
        # the blocked tool is absent from that record.
        command, seen = fake_server
        client_out = KeptStringIO()
        run_guard(
            McpGuard(safety_enforcer),
            command,
            client_in=io.StringIO(jsonrpc.encode(tool_call(1, "rm -rf /*"))),
            client_out=client_out,
        )
        assert "rm" not in seen.read_text()
        assert _lines(client_out)[0]["result"]["isError"] is True

    def test_guard_relays_a_full_session(self, fake_server, safety_enforcer) -> None:
        command, seen = fake_server
        client_out = KeptStringIO()
        session = "".join(
            jsonrpc.encode(m)
            for m in [
                {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
                tool_call(1, "Read", {"path": "a"}),
                tool_call(2, "rm -rf /*"),
                tool_call(3, "Write", {"path": "b"}),
            ]
        )
        run_guard(
            McpGuard(safety_enforcer),
            command,
            client_in=io.StringIO(session),
            client_out=client_out,
        )
        received = seen.read_text().split()
        assert received == ["initialize", "Read", "Write"]
        # Every request is answered exactly once, denied ones included.
        assert sorted(m["id"] for m in _lines(client_out)) == [0, 1, 2, 3]


class TestChildLifecycle:
    def test_a_server_that_ignores_eof_is_terminated_not_leaked(
        self, tmp_path, safety_enforcer
    ) -> None:
        # A downstream server that does not exit when its stdin closes would
        # otherwise be left running for the life of the client.
        script = tmp_path / "stubborn_server.py"
        script.write_text("import time\ntime.sleep(300)\n")
        code = run_guard(
            McpGuard(safety_enforcer),
            [sys.executable, str(script)],
            client_in=io.StringIO(""),
            client_out=KeptStringIO(),
        )
        assert code != 0  # terminated by signal rather than a clean exit


class TestCliSubprocess:
    def test_guard_runs_as_an_installed_console_script(self, fake_server) -> None:
        # Proves the entry point, argument parsing and `--` handling work as a
        # client would actually launch them.
        command, seen = fake_server
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "agentassert_abc.mcp.cli",
                "--contract",
                str(CONTRACTS / "safety-minimal.yaml"),
                "--server-label",
                "files",
                "--",
                *command,
            ],
            input=jsonrpc.encode(tool_call(1, "rm -rf /*"))
            + jsonrpc.encode(tool_call(2, "Read", {"path": "a"})),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "rm" not in seen.read_text()
        assert "Read" in seen.read_text()
        replies = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        by_id = {r["id"]: r for r in replies}
        assert by_id[1]["result"]["isError"] is True
        assert by_id[2]["result"]["content"][0]["text"] == "ran"
