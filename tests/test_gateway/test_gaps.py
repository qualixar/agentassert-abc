# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_coverage_gaps.py` +
`tests/test_100cover.py` — the portions exercising ported gateway code
(process_eval direct calls, exceptions, DecisionResult, SessionContext,
DriftReport). DSL parser/validator and typec's discarded soft-import shim
coverage is dropped (see `tests/test_gateway/__init__.py`).
"""

from __future__ import annotations

import json

import pytest

from agentassert_abc.exceptions import ContractBreachError, ContractLoadError
from agentassert_abc.gateway.compiler import CompiledContract
from agentassert_abc.gateway.events import (
    ContextWindow,
    DriftReport,
    HistoryDigest,
    PreAction,
    SessionContext,
    TurnEnd,
)
from agentassert_abc.gateway.process_eval import (
    evaluate_context_budget,
    evaluate_must_state,
    evaluate_turn_end_soft,
)
from agentassert_abc.gateway.violation_log import ViolationLog
from agentassert_abc.metrics.drift import DriftTracker
from agentassert_abc.metrics.theta import ThetaScorer
from agentassert_abc.models import DriftConfig
from agentassert_abc.process.models import (
    ContextBudget,
    ContractSpecExtended,
    DecisionResult,
    InvariantsExtended,
    MustState,
    ProcessDrift,
    ProcessInvariants,
    TypeCDecision,
)


class TestProcessEvalGaps:
    def test_must_state_deny(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="ms-deny",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    must_state=[
                        MustState(
                            field="cost",
                            before_tool_pattern="paid_api_*",
                            rationale="Must state cost",
                        )
                    ]
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        ctx = SessionContext(session_id="s1", stated_fields=frozenset({"other_field"}))
        event = PreAction(
            session_id="s1", contract_id="c1", tool="paid_api_deepseek", args={}, context=ctx
        )
        result = evaluate_must_state(event, compiled, ViolationLog())
        assert result is not None
        assert result.is_deny()
        assert result.violation_name == "must_state"

    def test_must_state_no_context_denies(self) -> None:
        """Fail-secure: no context = no evidence field was stated = DENY."""
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="ms-nocontext",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    must_state=[MustState(field="cost", before_tool_pattern="paid_api_*")]
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        event = PreAction(
            session_id="s1", contract_id="c1", tool="paid_api_deepseek", args={}, context=None
        )
        result = evaluate_must_state(event, compiled, ViolationLog())
        assert result is not None
        assert result.is_deny()

    def test_must_state_pattern_no_match(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="ms-nomatch",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    must_state=[MustState(field="cost", before_tool_pattern="paid_api_*")]
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        ctx = SessionContext(session_id="s1", stated_fields=frozenset())
        event = PreAction(session_id="s1", contract_id="c1", tool="Read", args={}, context=ctx)
        result = evaluate_must_state(event, compiled, ViolationLog())
        assert result is None

    def test_context_budget_compress(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="cb-compress",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    context_budget=ContextBudget(
                        max_tokens_per_turn=100, action_on_breach="compress"
                    )
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        event = ContextWindow(
            session_id="s1", contract_id="c1", token_count=500, prefix_hash="abc"
        )
        result = evaluate_context_budget(event, compiled, ViolationLog())
        assert result.decision == TypeCDecision.ALLOW
        assert result.reason == "compress_hint"

    def test_context_budget_within_limit(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="cb-ok",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(context_budget=ContextBudget(max_tokens_per_turn=100000))
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        event = ContextWindow(
            session_id="s1", contract_id="c1", token_count=500, prefix_hash="abc"
        )
        result = evaluate_context_budget(event, compiled, ViolationLog())
        assert result.decision == TypeCDecision.ALLOW

    def test_evaluate_turn_end_soft_no_config(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="no-drift",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(process=ProcessInvariants()),
        )
        compiled = CompiledContract.from_spec(spec)
        event = TurnEnd(session_id="s1", contract_id="c1", assistant_output="test")
        result = evaluate_turn_end_soft(
            event, compiled, DriftTracker(), ThetaScorer(), ViolationLog()
        )
        assert result.decision == TypeCDecision.ALLOW

    def test_evaluate_turn_end_soft_below_threshold(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="drift-below",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    process_drift=ProcessDrift(window_size=3, jsd_threshold=0.9, action="log")
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        drift = DriftTracker(config=DriftConfig(window=3))
        drift.compute_drift(c_total=1.0, action_dist={"A": 1.0})
        violations = ViolationLog()
        event = TurnEnd(session_id="s1", contract_id="c1", assistant_output="test")
        result = evaluate_turn_end_soft(event, compiled, drift, ThetaScorer(), violations)
        assert result.decision == TypeCDecision.ALLOW
        assert len(violations.all_violations()) == 0


class TestExceptions:
    def test_contract_breach_error_to_dict(self) -> None:
        e = ContractBreachError(
            violation_name="test", reason="bad", tool="rm", session_id="s1", contract_id="c1"
        )
        d = e.to_dict()
        assert d["violation_name"] == "test"
        assert d["decision"] == "deny"

    def test_contract_breach_error_to_json(self) -> None:
        e = ContractBreachError(
            violation_name="test", reason="bad", tool="rm", session_id="s1", contract_id="c1"
        )
        parsed = json.loads(e.to_json())
        assert parsed["violation_name"] == "test"

    def test_contract_breach_error_to_http_body(self) -> None:
        e = ContractBreachError(
            violation_name="test", reason="bad", tool="rm", session_id="s1", contract_id="c1"
        )
        body = e.to_http_body()
        assert body["error"] == "ContractBreachError"
        assert body["violation"] == "test"
        assert body["tool"] == "rm"

    def test_contract_load_error(self) -> None:
        with pytest.raises(ContractLoadError):
            raise ContractLoadError("bad contract")


class TestDecisionResult:
    def test_is_modify_true(self) -> None:
        dr = DecisionResult(decision=TypeCDecision.MODIFY)
        assert dr.is_modify()
        assert not dr.is_deny()

    def test_is_modify_false_for_allow(self) -> None:
        assert not DecisionResult(decision=TypeCDecision.ALLOW).is_modify()

    def test_is_modify_false_for_deny(self) -> None:
        assert not DecisionResult(decision=TypeCDecision.DENY).is_modify()

    def test_is_warn(self) -> None:
        assert DecisionResult(decision=TypeCDecision.WARN).is_warn()

    def test_is_redact(self) -> None:
        assert DecisionResult(decision=TypeCDecision.REDACT).is_redact()


class TestSessionContext:
    def test_has_stated_field_true(self) -> None:
        ctx = SessionContext(session_id="s1", stated_fields=frozenset({"cost", "reason"}))
        assert ctx.has_stated_field("cost")
        assert ctx.has_stated_field("reason")
        assert not ctx.has_stated_field("unknown")

    def test_session_context_defaults(self) -> None:
        ctx = SessionContext(session_id="s1")
        assert ctx.turn_index == 0
        assert ctx.token_count == 0


class TestHistoryDigestAndDriftReport:
    def test_history_digest(self) -> None:
        hd = HistoryDigest(turn_count=5, total_tokens=25000, role_pattern="U-A-U-A")
        assert hd.turn_count == 5
        assert hd.total_tokens == 25000

    def test_drift_report(self) -> None:
        dr = DriftReport(current_jsd=0.15, window_size=10, violation_count=2)
        assert dr.current_jsd == 0.15
        assert dr.violation_count == 2
