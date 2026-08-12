# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_session_judge.py` +
`tests/test_operator_enforcement.py` +
`tests/test_evaluator_engine.py::TestSessionMonitor` (renamed
SessionMonitor -> SessionEnforcer per the migration notes — FATAL name collision
with abc v2's existing, unrelated `SessionMonitor`).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from agentassert_abc.exceptions import ContractLoadError
from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.gateway.events import PreAction, TurnEnd, TurnStart
from agentassert_abc.process.models import (
    ContractSpecExtended,
    InvariantsExtended,
    JudgePredicate,
    MustPrecede,
    ProcessInvariants,
    ToolAllowlist,
    ToolBlocklist,
    TypeCDecision,
)

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


def _make_enforcer(invariants_data: dict) -> SessionEnforcer:
    contract = ContractSpecExtended(
        contractspec="1.0",
        kind="agent",
        name="test",
        description="test",
        version="0.1",
        invariants=InvariantsExtended(process=ProcessInvariants(**invariants_data)),
    )
    return SessionEnforcer(contract)


class TestJudgeDispatcherWiring:
    def test_init_creates_judge_dispatchers(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="judge-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    judge_predicate=[
                        JudgePredicate(rubric="Test rubric", sample_rate=0.3, model="haiku"),
                        JudgePredicate(rubric="Another", sample_rate=0.1, model="free-tier-model"),
                    ]
                )
            ),
        )
        enforcer = SessionEnforcer(spec)
        assert len(enforcer._judge_dispatchers) == 2

    def test_no_process_invariants_no_dispatchers(self) -> None:
        spec = ContractSpecExtended(
            contractspec="1.0", kind="agent", name="simple", description="test", version="0.1"
        )
        enforcer = SessionEnforcer(spec)
        assert len(enforcer._judge_dispatchers) == 0

    def test_schedule_judge_evaluation_samples(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="judge-sample",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    judge_predicate=[JudgePredicate(rubric="Test", sample_rate=1.0, model="haiku")]
                )
            ),
        )
        enforcer = SessionEnforcer(spec)
        assert len(enforcer._judge_dispatchers) == 1
        enforcer.schedule_judge_evaluation("output", "s1")
        assert enforcer._judge_dispatchers[0]._sample_count >= 1

    def test_no_invariants_early_return(self) -> None:
        spec = ContractSpecExtended(
            contractspec="1.0", kind="agent", name="no-inv", description="test", version="0.1"
        )
        enforcer = SessionEnforcer(spec)
        enforcer.schedule_judge_evaluation("output", "s1")  # must not raise

    def test_invariants_no_process_early_return(self) -> None:
        from agentassert_abc.models import ConstraintCheck, HardConstraint

        spec = ContractSpecExtended(
            contractspec="1.0",
            kind="agent",
            name="hard-only",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                hard=[
                    HardConstraint(
                        name="pii-check", check=ConstraintCheck(field="pii", equals=False)
                    )
                ]
            ),
        )
        enforcer = SessionEnforcer(spec)
        enforcer.schedule_judge_evaluation("output", "s1")  # must not raise


class TestTurnAndDenyCounts:
    def test_turn_count_incremented_on_turnend(self) -> None:
        spec = ContractSpecExtended(
            contractspec="1.0", kind="agent", name="turn-test", description="test", version="0.1"
        )
        enforcer = SessionEnforcer(spec)
        assert enforcer.turn_count == 0
        enforcer.evaluate(TurnEnd(session_id="s1", contract_id="c1", assistant_output="hello"))
        assert enforcer.turn_count == 1

    def test_deny_count_incremented(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="deny-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(tool_blocklist=[ToolBlocklist(tools=["rm"])])
            ),
        )
        enforcer = SessionEnforcer(spec)
        assert enforcer.deny_count == 0
        result = enforcer.evaluate(
            PreAction(session_id="s1", contract_id="c1", tool="rm", args={})
        )
        assert result.is_deny()
        assert enforcer.deny_count == 1
        # Additive fix beyond typec parity: a DENY also feeds ThetaScorer's
        # event-frequency term (typec never called record_violation() from
        # its production dispatch path — see gateway/enforcer.py docstring).
        assert enforcer._theta._violation_count == 1


class TestToolAllowlist:
    def test_allowed_tool_passes(self) -> None:
        enforcer = _make_enforcer(
            {"tool_allowlist": [ToolAllowlist(tools=["read_file", "write_file"])]}
        )
        result = enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="read_file"))
        assert result.decision == TypeCDecision.ALLOW

    def test_wildcard_allowed(self) -> None:
        enforcer = _make_enforcer({"tool_allowlist": [ToolAllowlist(tools=["bash_*"])]})
        result = enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="bash_run"))
        assert result.decision == TypeCDecision.ALLOW

    def test_blocked_tool_denied(self) -> None:
        enforcer = _make_enforcer({"tool_allowlist": [ToolAllowlist(tools=["read_file"])]})
        result = enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="bash_run"))
        assert result.decision == TypeCDecision.DENY
        assert "tool_allowlist" in result.violation_name

    def test_empty_allowlist_allows_all(self) -> None:
        enforcer = _make_enforcer({})
        result = enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="anything"))
        assert result.decision == TypeCDecision.ALLOW

    def test_multi_block_union_allows(self) -> None:
        enforcer = _make_enforcer(
            {
                "tool_allowlist": [
                    ToolAllowlist(tools=["read_file"]),
                    ToolAllowlist(tools=["write_file"]),
                ]
            }
        )
        for tool in ("read_file", "write_file"):
            result = enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool=tool))
            assert result.decision == TypeCDecision.ALLOW

    def test_multi_block_blocked_if_in_none(self) -> None:
        enforcer = _make_enforcer(
            {
                "tool_allowlist": [
                    ToolAllowlist(tools=["read_file"]),
                    ToolAllowlist(tools=["write_file"]),
                ]
            }
        )
        result = enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="bash"))
        assert result.decision == TypeCDecision.DENY


class TestMustPrecede:
    def test_correct_order_passes(self) -> None:
        enforcer = _make_enforcer({"must_precede": [MustPrecede(before="plan", after="execute")]})
        enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="plan"))
        result = enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="execute"))
        assert result.decision == TypeCDecision.ALLOW

    def test_wrong_order_denied(self) -> None:
        enforcer = _make_enforcer({"must_precede": [MustPrecede(before="plan", after="execute")]})
        result = enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="execute"))
        assert result.decision == TypeCDecision.DENY
        assert "must_precede" in result.violation_name

    def test_unrelated_tool_not_blocked(self) -> None:
        enforcer = _make_enforcer({"must_precede": [MustPrecede(before="plan", after="execute")]})
        result = enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="read_file"))
        assert result.decision == TypeCDecision.ALLOW

    def test_turn_scope_resets_on_turn_end(self) -> None:
        enforcer = _make_enforcer(
            {"must_precede": [MustPrecede(before="plan", after="execute", scope="turn")]}
        )
        enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="plan"))
        enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="execute"))
        enforcer.evaluate(TurnEnd(session_id="s", contract_id="test", assistant_output=""))
        result = enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="execute"))
        assert result.decision == TypeCDecision.DENY

    def test_session_scope_persists_across_turns(self) -> None:
        enforcer = _make_enforcer(
            {"must_precede": [MustPrecede(before="plan", after="execute", scope="session")]}
        )
        enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="plan"))
        enforcer.evaluate(TurnEnd(session_id="s", contract_id="test", assistant_output=""))
        result = enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="execute"))
        assert result.decision == TypeCDecision.ALLOW


class TestTurnEndSeenToolsReset:
    def test_seen_tools_turn_cleared_on_turn_end(self) -> None:
        enforcer = _make_enforcer(
            {"must_precede": [MustPrecede(before="plan", after="execute", scope="turn")]}
        )
        enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="plan"))
        assert "plan" in enforcer._seen_tools_turn
        enforcer.evaluate(TurnEnd(session_id="s", contract_id="test", assistant_output=""))
        assert "plan" not in enforcer._seen_tools_turn

    def test_seen_tools_session_not_cleared_on_turn_end(self) -> None:
        enforcer = _make_enforcer({})
        enforcer.evaluate(PreAction(session_id="s", contract_id="test", tool="read_file"))
        enforcer.evaluate(TurnEnd(session_id="s", contract_id="test", assistant_output=""))
        assert "read_file" in enforcer._seen_tools_session


class TestFromYaml:
    def test_from_yaml_valid(self) -> None:
        enforcer = SessionEnforcer.from_yaml(str(FIXTURES / "safety-minimal.yaml"))
        assert enforcer is not None

    def test_from_yaml_invalid(self) -> None:
        with pytest.raises(ContractLoadError):
            SessionEnforcer.from_yaml(str(FIXTURES / "invalid-missing-name.yaml"))

    def test_abc_compat_contract(self) -> None:
        enforcer = SessionEnforcer.from_yaml(str(FIXTURES / "abc-v03-compat.yaml"))
        result = enforcer.evaluate(
            PreAction(session_id="s1", contract_id="c1", tool="Read", args={})
        )
        assert result.decision == TypeCDecision.ALLOW

    def test_full_governance_contract_loads_all_operators(self) -> None:
        """All 7 process operators + hard invariant parse and compile together."""
        enforcer = SessionEnforcer.from_yaml(str(FIXTURES / "full-governance.yaml"))
        proc = enforcer._contract.invariants.process
        assert len(proc.tool_blocklist) == 1
        assert len(proc.tool_allowlist) == 1
        assert len(proc.must_state) == 1
        assert len(proc.must_precede) == 1
        assert proc.context_budget is not None
        assert proc.process_drift is not None
        assert len(proc.judge_predicate) == 1
        # tool_blocklist should deny "rm -rf /*"
        result = enforcer.evaluate(PreAction(session_id="s1", contract_id="c1", tool="rm -rf /*"))
        assert result.is_deny()


class TestConcurrencyAndClose:
    def test_concurrent_evaluate(self) -> None:
        spec = ContractSpecExtended(
            contractspec="1.0", kind="agent", name="empty", description="empty", version="0.1"
        )
        enforcer = SessionEnforcer(spec)
        results: list[object] = []
        errors: list[Exception] = []

        def make_call() -> None:
            try:
                results.append(
                    enforcer.evaluate(
                        PreAction(session_id="s1", contract_id="c1", tool="Read", args={})
                    )
                )
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=make_call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(r.decision == TypeCDecision.ALLOW for r in results)  # type: ignore[attr-defined]

    def test_close_returns_session_end(self) -> None:
        spec = ContractSpecExtended(
            contractspec="1.0", kind="agent", name="empty", description="empty", version="0.1"
        )
        enforcer = SessionEnforcer(spec)
        result = enforcer.close()
        assert result.theta > 0

    def test_unknown_event_type_allows(self) -> None:
        spec = ContractSpecExtended(
            contractspec="1.0", kind="agent", name="empty", description="empty", version="0.1"
        )
        enforcer = SessionEnforcer(spec)
        result = enforcer.evaluate(
            TurnStart(session_id="s1", contract_id="c1", user_input="hello")
        )
        assert result.decision == TypeCDecision.ALLOW
