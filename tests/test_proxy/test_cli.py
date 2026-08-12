# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for the agentassert-abc-proxy CLI.

Invariants pinned here:
  * `start` with a non-existent contract exits with code 1 and emits an
    error message — the proxy must never silently start with a bad contract.
  * `start` with a valid contract boots (monkeypatched uvicorn.run) and
    emits the listening address without binding an actual port.
  * `status` with a reachable proxy prints the response body and exits 0.
  * `status` when the proxy is unreachable exits with code 1 and prints an
    error to stderr.
  * `_warn_if_upstream_mismatch` emits a warning when env vars point to a
    non-default backend but the contract has no upstream block, and stays
    silent when the contract DOES have an upstream block.

All tests use `click.testing.CliRunner` — no ports are bound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from pathlib import Path
from typing import Any

from click.testing import CliRunner

from agentassert_abc.proxy.cli import _warn_if_upstream_mismatch, cli

FIXTURES = Path(__file__).parent.parent / "test_gateway" / "fixtures" / "contracts"


# ---------------------------------------------------------------------------
# start command — missing contract
# ---------------------------------------------------------------------------


class TestStartMissingContract:
    def test_missing_contract_exits_nonzero(self, tmp_path: Path) -> None:
        """start with a path that does not exist must exit 1 immediately.

        This guards the invariant that the proxy never silently starts
        without a validated contract file.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["start", "--contract", str(tmp_path / "no-such-file.yaml")],
        )
        assert result.exit_code == 1

    def test_missing_contract_shows_error_message(self, tmp_path: Path) -> None:
        """Error output must mention 'Contract file not found' for debuggability.

        Click 8.4.2 mixes stderr into result.output by default.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["start", "--contract", str(tmp_path / "ghost.yaml")],
        )
        # Click 8.4.2: err=True output is also in result.output.
        assert "Contract file not found" in result.output


# ---------------------------------------------------------------------------
# start command — valid contract, uvicorn monkeypatched
# ---------------------------------------------------------------------------


class TestStartValidContract:
    def test_start_emits_listen_address(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """start with a real contract must print the listening address.

        uvicorn.run is monkeypatched to avoid actually binding a port.
        create_app is also monkeypatched to avoid DB initialisation.
        """
        captured_runs: list[dict[str, Any]] = []

        def _fake_uvicorn_run(app: Any, *, host: str, port: int, **_: Any) -> None:
            captured_runs.append({"host": host, "port": port})

        def _fake_create_app(**_: Any) -> object:
            return object()

        monkeypatch.setattr("agentassert_abc.proxy.cli.uvicorn.run", _fake_uvicorn_run)
        monkeypatch.setattr(
            "agentassert_abc.proxy.server.create_app",
            _fake_create_app,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "start",
                "--contract", str(FIXTURES / "safety-minimal.yaml"),
                "--port", "19000",
                "--host", "127.0.0.1",
            ],
        )
        assert "Listening on http://127.0.0.1:19000" in result.output
        assert len(captured_runs) == 1
        assert captured_runs[0]["port"] == 19000

    def test_start_no_persist_flag_emits_notice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--no-persist must produce 'Persistence: disabled' in the banner."""
        monkeypatch.setattr(
            "agentassert_abc.proxy.cli.uvicorn.run",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "agentassert_abc.proxy.server.create_app",
            lambda **kw: object(),
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "start",
                "--contract", str(FIXTURES / "safety-minimal.yaml"),
                "--no-persist",
            ],
        )
        assert "Persistence: disabled" in result.output

    def test_start_default_session_id_banner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without --session-id, banner must show 'session_id=default'."""
        monkeypatch.setattr(
            "agentassert_abc.proxy.cli.uvicorn.run",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "agentassert_abc.proxy.server.create_app",
            lambda **kw: object(),
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "start",
                "--contract", str(FIXTURES / "safety-minimal.yaml"),
            ],
        )
        assert "session_id=default" in result.output


# ---------------------------------------------------------------------------
# status command — reachable proxy
# ---------------------------------------------------------------------------


class TestStatusReachable:
    def test_status_prints_response_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """status with a reachable proxy must echo the response body and exit 0."""
        import httpx

        class _FakeResponse:
            text = '{"status": "healthy"}'

        monkeypatch.setattr(
            httpx, "get", lambda url, **kw: _FakeResponse()
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--port", "9000"])
        assert result.exit_code == 0
        assert "healthy" in result.output


# ---------------------------------------------------------------------------
# status command — unreachable proxy
# ---------------------------------------------------------------------------


class TestStatusUnreachable:
    def test_status_http_error_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """status when proxy is unreachable must exit 1 and print error."""
        import httpx

        def _raise(*_: Any, **__: Any) -> None:
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "get", _raise)

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--port", "9999"], mix_stderr=False)
        assert result.exit_code == 1

    def test_status_http_error_shows_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error message must contain 'Proxy not reachable'.

        Click 8.4.2 mixes err=True output into result.output.
        """
        import httpx

        def _raise(*_: Any, **__: Any) -> None:
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", _raise)

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])
        # Click 8.4.2: err=True output mixed into result.output.
        assert "Proxy not reachable" in result.output


# ---------------------------------------------------------------------------
# _warn_if_upstream_mismatch — env var logic
# ---------------------------------------------------------------------------


class TestWarnIfUpstreamMismatch:
    def test_no_warning_when_env_vars_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No warning when no custom base URL env vars are set.

        Call _warn_if_upstream_mismatch() directly (no env vars) and confirm
        no warning text is emitted. This avoids Click version differences.
        """
        import io

        contract = tmp_path / "no-upstream.yaml"
        contract.write_text(
            "contractspec: '0.1'\nkind: agent\nname: t\ndescription: t\nversion: '1.0.0'\n"
        )
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        buf = io.StringIO()

        def _capture(txt: str = "", **_: Any) -> None:
            buf.write(txt + "\n")

        monkeypatch.setattr("agentassert_abc.proxy.cli.click.echo", _capture)
        _warn_if_upstream_mismatch(contract)
        assert "Warning: non-default upstream" not in buf.getvalue()

    def test_warning_when_anthropic_env_set_no_upstream_in_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Warning fires when ANTHROPIC_BASE_URL is non-default and contract
        lacks an 'upstream:' block.
        """
        contract = tmp_path / "no-upstream.yaml"
        contract.write_text(
            "contractspec: '0.1'\nkind: agent\nname: t\ndescription: t\nversion: '1.0.0'\n"
        )
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.proxy.example.com")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        import io

        buf = io.StringIO()

        def _capture(txt: str = "", **_: Any) -> None:
            buf.write(txt + "\n")

        monkeypatch.setattr("agentassert_abc.proxy.cli.click.echo", _capture)
        _warn_if_upstream_mismatch(contract)
        assert "Warning: non-default upstream" in buf.getvalue()

    def test_no_warning_when_contract_has_upstream_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No warning when the contract already has an 'upstream:' block,
        even if ANTHROPIC_BASE_URL is set to a non-default URL.
        """
        contract = tmp_path / "with-upstream.yaml"
        contract.write_text(
            "contractspec: '0.1'\nkind: agent\nname: t\ndescription: t\n"
            "version: '1.0.0'\nupstream:\n  anthropic: https://api.anthropic.com\n"
        )
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.proxy.example.com")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        import io

        buf = io.StringIO()

        def _capture(txt: str = "", **_: Any) -> None:
            buf.write(txt + "\n")

        monkeypatch.setattr("agentassert_abc.proxy.cli.click.echo", _capture)
        _warn_if_upstream_mismatch(contract)
        assert "Warning" not in buf.getvalue()

    def test_warn_mismatch_on_bad_yaml_is_silent(self, tmp_path: Path) -> None:
        """If the YAML is malformed, _warn_if_upstream_mismatch must return
        without raising — it swallows the exception silently.
        """
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{unclosed bracket: [")
        # Must not raise.
        _warn_if_upstream_mismatch(bad_yaml)
