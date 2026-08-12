# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""The Claude Code hook — a shipped adoption surface that had no tests.

It is fail-open by design: any bad input, unloadable contract or evaluation
error must let the tool call through rather than wedge the user's session. That
property is what these tests pin, alongside the enforcement it does perform.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from agentassert_abc.claude_code import hook as hook_mod

from ..test_mcp.conftest import CONTRACTS


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path: Path):
    """Never let this module read a stale cache or touch the real home dir."""
    hook_mod._enforcer_cache.clear()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("AGENTASSERT_CONTRACT", raising=False)
    yield
    hook_mod._enforcer_cache.clear()


def run_hook(payload: Any, monkeypatch, capsys, contract: str | None = None) -> dict[str, Any]:
    """Drive ``hook.main()`` the way Claude Code does: JSON on stdin."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(raw))
    if contract is not None:
        monkeypatch.setenv("AGENTASSERT_CONTRACT", contract)
    hook_mod.main()
    return json.loads(capsys.readouterr().out)


SAFETY = str(CONTRACTS / "safety-minimal.yaml")


class TestEnforcement:
    def test_blocklisted_tool_is_blocked(self, monkeypatch, capsys) -> None:
        out = run_hook(
            {"hook_event_name": "PreToolUse", "tool_name": "rm -rf /*", "tool_input": {}},
            monkeypatch,
            capsys,
            contract=SAFETY,
        )
        assert out["action"] == "block"
        assert out["violation"]
        assert out["reason"]

    def test_permitted_tool_is_allowed(self, monkeypatch, capsys) -> None:
        out = run_hook(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Read",
                "tool_input": {"file_path": "a.txt"},
            },
            monkeypatch,
            capsys,
            contract=SAFETY,
        )
        assert out["action"] == "allow"

    def test_post_tool_use_is_scored_and_allowed(self, monkeypatch, capsys) -> None:
        out = run_hook(
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Read",
                "tool_output": {"content": "hello"},
            },
            monkeypatch,
            capsys,
            contract=SAFETY,
        )
        assert out["action"] == "allow"

    def test_post_tool_output_reaches_the_contract_as_flat_state(self, monkeypatch) -> None:
        # The 0.6.0 defect: passing no state scored every semantic invariant as
        # a violation. Assert the `output.*` convention actually arrives.
        from agentassert_abc.gateway.events import PostAction

        event = hook_mod._event_from_hook(
            "PostToolUse",
            {"tool_name": "Read", "tool_output": {"pii_detected": False, "u": {"t": 7}}},
            "s1",
            "c1",
        )
        assert isinstance(event, PostAction)
        assert event.state["output.pii_detected"] is False
        assert event.state["output.u.t"] == 7
        assert event.state["tool.name"] == "Read"

    def test_modify_decision_returns_rewritten_tool_input(self, monkeypatch, capsys) -> None:
        # Claude Code applies `tool_input` from the hook's reply, so this is
        # what makes a MODIFY verdict actually change the call.
        import agentassert_abc.gateway.enforcer as enf
        from agentassert_abc.process.models import DecisionResult, TypeCDecision

        monkeypatch.setattr(
            enf.SessionEnforcer,
            "evaluate",
            lambda *_a, **_k: DecisionResult(
                decision=TypeCDecision.MODIFY, modified_args={"file_path": "/tmp/safe"}
            ),
        )
        out = run_hook(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Read",
                "tool_input": {"file_path": "/etc/passwd"},
            },
            monkeypatch,
            capsys,
            contract=SAFETY,
        )
        assert out["action"] == "modify"
        assert out["tool_input"] == {"file_path": "/tmp/safe"}

    def test_tool_name_falls_back_to_the_nested_input_key(self) -> None:
        event = hook_mod._event_from_hook(
            "PreToolUse", {"tool_name_input": {"tool_name": "Bash"}}, "s", "c"
        )
        assert event.tool == "Bash"

    def test_legacy_hook_type_key_is_understood(self, monkeypatch, capsys) -> None:
        out = run_hook(
            {"hook_type": "PreToolUse", "tool_name": "rm -rf /*", "tool_input": {}},
            monkeypatch,
            capsys,
            contract=SAFETY,
        )
        assert out["action"] == "block"


class TestFailOpen:
    """Every one of these must allow — the hook must never wedge a session."""

    def test_unparseable_stdin_allows(self, monkeypatch, capsys) -> None:
        assert run_hook("not json at all", monkeypatch, capsys)["action"] == "allow"

    def test_empty_stdin_allows(self, monkeypatch, capsys) -> None:
        assert run_hook("", monkeypatch, capsys)["action"] == "allow"

    def test_no_contract_configured_allows(self, monkeypatch, capsys) -> None:
        out = run_hook(
            {"hook_event_name": "PreToolUse", "tool_name": "rm -rf /*"}, monkeypatch, capsys
        )
        assert out["action"] == "allow"

    def test_missing_contract_file_allows(self, monkeypatch, capsys) -> None:
        out = run_hook(
            {"hook_event_name": "PreToolUse", "tool_name": "rm -rf /*"},
            monkeypatch,
            capsys,
            contract="/nonexistent/contract.yaml",
        )
        assert out["action"] == "allow"

    def test_unknown_event_type_allows(self, monkeypatch, capsys) -> None:
        out = run_hook(
            {"hook_event_name": "SomethingElse", "tool_name": "rm -rf /*"},
            monkeypatch,
            capsys,
            contract=SAFETY,
        )
        assert out["action"] == "allow"

    def test_evaluation_error_allows(self, monkeypatch, capsys) -> None:
        import agentassert_abc.gateway.enforcer as enf

        monkeypatch.setattr(
            enf.SessionEnforcer,
            "evaluate",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        out = run_hook(
            {"hook_event_name": "PreToolUse", "tool_name": "rm -rf /*"},
            monkeypatch,
            capsys,
            contract=SAFETY,
        )
        assert out["action"] == "allow"


class TestContractGate:
    def test_unevaluable_contract_is_refused_with_a_reason_on_stderr(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        # Still fail-open (it must not block the session), but it says why
        # instead of silently enforcing nothing.
        contract = tmp_path / "unusable.yaml"
        contract.write_text(
            "dsl_version: '0.4'\ncontractspec: '1.0'\nkind: agent\n"
            "name: unusable\ndescription: 'needs unseen state'\nversion: '0.1'\n"
            "invariants:\n  hard:\n    - name: needs-db\n"
            "      description: 'no rows written'\n"
            "      check:\n        field: database.rows_written\n        equals: 0\n"
        )
        monkeypatch.setattr(
            "sys.stdin", io.StringIO(json.dumps({"hook_event_name": "PreToolUse"}))
        )
        monkeypatch.setenv("AGENTASSERT_CONTRACT", str(contract))
        hook_mod.main()
        captured = capsys.readouterr()
        assert json.loads(captured.out)["action"] == "allow"
        assert "database.rows_written" in captured.err

    def test_enforcer_is_cached_across_calls(self, monkeypatch, capsys) -> None:
        # The hook is spawned per tool call in some setups but reused in others;
        # reloading the contract every time would be a per-call YAML parse.
        run_hook(
            {"hook_event_name": "PreToolUse", "tool_name": "Read"},
            monkeypatch,
            capsys,
            contract=SAFETY,
        )
        assert SAFETY in hook_mod._enforcer_cache
        first = hook_mod._enforcer_cache[SAFETY]
        run_hook(
            {"hook_event_name": "PreToolUse", "tool_name": "Read"},
            monkeypatch,
            capsys,
            contract=SAFETY,
        )
        assert hook_mod._enforcer_cache[SAFETY] is first

    def test_empty_contract_path_yields_no_enforcer(self) -> None:
        assert hook_mod._get_enforcer("") is None
