# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_evaluator_engine.py` +
`tests/test_coverage_gaps.py::TestEvaluatorEngineGaps` +
`tests/test_precise_gaps.py`'s dispatch_event coverage.

`DriftTracker(window=N)` + `.update(tool=...)` fixtures (typec's discarded
tracker API) are replaced with `compute_drift(c_total=..., action_dist=...)`
calls against abc v2's `DriftTracker` (the migration notes) — see `_pump_drift`.
"""

from __future__ import annotations

from agentassert_abc.gateway.compiler import CompiledContract
from agentassert_abc.gateway.engine import dispatch_event
from agentassert_abc.gateway.events import (
    ContextWindow,
    PostAction,
    PreAction,
    SessionEnd,
    SessionStart,
    TurnEnd,
    TurnStart,
    TypeCEvent,
)
from agentassert_abc.gateway.violation_log import ViolationLog
from agentassert_abc.metrics.drift import DriftTracker
from agentassert_abc.metrics.theta import ThetaScorer
from agentassert_abc.models import DriftConfig
from agentassert_abc.process.models import (
    ContextBudget,
    ContractSpecExtended,
    InvariantsExtended,
    MustState,
    ProcessDrift,
    ProcessInvariants,
    ToolBlocklist,
    TypeCDecision,
)


def _empty_spec() -> ContractSpecExtended:
    return ContractSpecExtended(
        contractspec="1.0", kind="agent", name="empty", description="empty", version="0.1"
    )


def _pump_drift(drift: DriftTracker, tool: str, c_total: float, times: int) -> None:
    """Repeatedly record a turn's D(t) — the abc-v2 replacement for typec's
    `drift.update(tool=...)`. A low `c_total` reliably drives D(t) above any
    reasonable threshold via the compliance term alone; `c_total=1.0` with a
    single tool label keeps D(t) at 0.
    """
    for _ in range(times):
        drift.compute_drift(c_total=c_total, action_dist={tool: 1.0})


class TestPreActionGates:
    def test_pre_action_blocked_tool(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="blocklist-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    tool_blocklist=[ToolBlocklist(tools=["rm", "curl|bash"])]
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        result = dispatch_event(
            PreAction(session_id="s1", contract_id="c1", tool="rm", args={}),
            compiled,
            DriftTracker(),
            ThetaScorer(),
            ViolationLog(),
        )
        assert result.is_deny()
        assert result.violation_name == "tool_blocklist"

    def test_pre_action_allowed_tool(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="blocklist-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    tool_blocklist=[ToolBlocklist(tools=["rm", "curl|bash"])]
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        result = dispatch_event(
            PreAction(session_id="s1", contract_id="c1", tool="Read", args={}),
            compiled,
            DriftTracker(),
            ThetaScorer(),
            ViolationLog(),
        )
        assert result.decision == TypeCDecision.ALLOW

    def test_pre_action_blocklist_or_pattern(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="blocklist-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    tool_blocklist=[ToolBlocklist(tools=["rm", "curl|bash"])]
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        for tool in ["curl", "bash", "curl some args"]:
            result = dispatch_event(
                PreAction(session_id="s1", contract_id="c1", tool=tool, args={}),
                compiled,
                DriftTracker(),
                ThetaScorer(),
                ViolationLog(),
            )
            assert result.is_deny(), f"tool '{tool}' should be blocked"

    def test_must_state_deny_through_dispatch_event(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="ms-dispatch",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    tool_blocklist=[ToolBlocklist(tools=["zzz"])],
                    must_state=[
                        MustState(
                            field="cost",
                            before_tool_pattern="paid_api_*",
                            rationale="Must state cost",
                        )
                    ],
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        from agentassert_abc.gateway.events import SessionContext

        ctx = SessionContext(session_id="s1", stated_fields=frozenset({"other"}))
        event = PreAction(
            session_id="s1", contract_id="c1", tool="paid_api_deepseek", args={}, context=ctx
        )
        result = dispatch_event(event, compiled, DriftTracker(), ThetaScorer(), ViolationLog())
        assert result.is_deny()
        assert result.violation_name == "must_state"


class TestPostActionCompliance:
    def test_post_action_allows_and_updates_drift(self) -> None:
        spec = ContractSpecExtended(
            contractspec="1.0", kind="agent", name="test", description="test", version="0.1"
        )
        compiled = CompiledContract.from_spec(spec)
        drift = DriftTracker(config=DriftConfig(window=50))
        theta = ThetaScorer()

        event = PostAction(
            session_id="s1",
            contract_id="c1",
            tool="Read",
            args={"path": "/tmp"},
            state={"bytes": 100},
        )
        result = dispatch_event(event, compiled, drift, theta, ViolationLog())
        assert result.decision == TypeCDecision.ALLOW
        # No hard/soft invariants configured => c_hard=c_soft=1.0 (real evaluation,
        # not a stub) and D(t) recorded (non-empty history — migration review).
        assert theta._compliance_scores == [1.0]
        assert len(drift.history) == 1

    def test_post_action_feeds_real_compliance_into_theta(self) -> None:
        """CRIT check (b): a violated hard invariant lowers c_hard fed to theta."""
        from agentassert_abc.models import ConstraintCheck, HardConstraint

        spec = ContractSpecExtended(
            contractspec="1.0",
            kind="agent",
            name="test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                hard=[
                    HardConstraint(name="no_pii", check=ConstraintCheck(field="pii", equals=False))
                ]
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        drift = DriftTracker(config=DriftConfig(window=50))
        theta = ThetaScorer()

        event = PostAction(session_id="s1", contract_id="c1", tool="search", state={"pii": True})
        dispatch_event(event, compiled, drift, theta, ViolationLog())

        assert theta._compliance_scores == [0.5]  # (c_hard=0.0 + c_soft=1.0) / 2
        assert drift.history[-1] > 0.0  # CRIT check (a): drift is non-zero, not stubbed


class TestContextWindow:
    def test_context_window_deny(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="budget-deny",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    context_budget=ContextBudget(max_tokens_per_turn=100, action_on_breach="deny")
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        event = ContextWindow(
            session_id="s1", contract_id="c1", token_count=500, prefix_hash="abc"
        )
        result = dispatch_event(event, compiled, DriftTracker(), ThetaScorer(), ViolationLog())
        assert result.is_deny()
        assert result.violation_name == "context_budget"

    def test_context_window_warn(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="budget-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    context_budget=ContextBudget(max_tokens_per_turn=100, action_on_breach="warn")
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        violations = ViolationLog()
        event = ContextWindow(
            session_id="s1", contract_id="c1", token_count=500, prefix_hash="abc"
        )
        result = dispatch_event(event, compiled, DriftTracker(), ThetaScorer(), violations)
        assert result.decision == TypeCDecision.ALLOW
        assert len(violations.all_violations()) == 1
        assert violations.all_violations()[0]["kind"] == "soft"


class TestOtherEventTypes:
    def test_session_start_dispatch(self) -> None:
        compiled = CompiledContract.from_spec(_empty_spec())
        event = SessionStart(session_id="s1", contract_id="c1", workdir="/tmp", model="claude")
        result = dispatch_event(event, compiled, DriftTracker(), ThetaScorer(), ViolationLog())
        assert result.decision == TypeCDecision.ALLOW
        assert result.reason == "session started"

    def test_session_end_dispatch(self) -> None:
        compiled = CompiledContract.from_spec(_empty_spec())
        event = SessionEnd(session_id="s1", contract_id="c1", theta=0.95)
        result = dispatch_event(event, compiled, DriftTracker(), ThetaScorer(), ViolationLog())
        assert result.decision == TypeCDecision.ALLOW
        assert result.reason == "session ended"

    def test_unknown_event_type_allows(self) -> None:
        compiled = CompiledContract.from_spec(_empty_spec())
        event = TypeCEvent(session_id="s1", contract_id="c1")
        result = dispatch_event(event, compiled, DriftTracker(), ThetaScorer(), ViolationLog())
        assert result.decision == TypeCDecision.ALLOW

    def test_turn_start_allows(self) -> None:
        compiled = CompiledContract.from_spec(_empty_spec())
        event = TurnStart(session_id="s1", contract_id="c1", user_input="hello")
        result = dispatch_event(event, compiled, DriftTracker(), ThetaScorer(), ViolationLog())
        assert result.decision == TypeCDecision.ALLOW


class TestTurnEndProcessDrift:
    def test_turn_end_with_drift_config_log(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="drift-log",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    process_drift=ProcessDrift(window_size=3, jsd_threshold=0.1, action="log")
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        drift = DriftTracker(config=DriftConfig(window=3))
        _pump_drift(drift, "A", c_total=0.0, times=5)
        violations = ViolationLog()
        event = TurnEnd(session_id="s1", contract_id="c1", assistant_output="test")
        result = dispatch_event(event, compiled, drift, ThetaScorer(), violations)
        assert result.decision == TypeCDecision.ALLOW
        assert len(violations.all_violations()) >= 1

    def test_turn_end_with_drift_config_warn(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="drift-warn",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    process_drift=ProcessDrift(window_size=3, jsd_threshold=0.1, action="warn")
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        drift = DriftTracker(config=DriftConfig(window=3))
        _pump_drift(drift, "A", c_total=0.0, times=5)
        violations = ViolationLog()
        event = TurnEnd(session_id="s1", contract_id="c1", assistant_output="test")
        result = dispatch_event(event, compiled, drift, ThetaScorer(), violations)
        assert result.decision == TypeCDecision.ALLOW
        assert len(violations.all_violations()) >= 1

    def test_turn_end_with_drift_config_theta_penalty(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="drift-penalty",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    process_drift=ProcessDrift(
                        window_size=3, jsd_threshold=0.1, action="theta_penalty"
                    )
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        drift = DriftTracker(config=DriftConfig(window=3))
        _pump_drift(drift, "A", c_total=0.0, times=5)
        theta = ThetaScorer()
        event = TurnEnd(session_id="s1", contract_id="c1", assistant_output="test")
        result = dispatch_event(event, compiled, drift, theta, ViolationLog())
        assert result.decision == TypeCDecision.ALLOW
        assert theta.compute() < 1.0

    def test_turn_end_no_drift_config(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="cb-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    context_budget=ContextBudget(
                        max_tokens_per_turn=10000, action_on_breach="warn"
                    )
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        event = TurnEnd(session_id="s1", contract_id="c1", assistant_output="test")
        result = dispatch_event(event, compiled, DriftTracker(), ThetaScorer(), ViolationLog())
        assert result.decision == TypeCDecision.ALLOW

    def test_drift_below_threshold_no_action(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="drift-low",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    process_drift=ProcessDrift(window_size=5, jsd_threshold=0.9, action="log")
                )
            ),
        )
        compiled = CompiledContract.from_spec(spec)
        drift = DriftTracker(config=DriftConfig(window=5))
        _pump_drift(drift, "OnlyTool", c_total=1.0, times=10)
        assert drift.history[-1] < 0.5
        result = dispatch_event(
            TurnEnd(session_id="s1", contract_id="c1", assistant_output="x"),
            compiled,
            drift,
            ThetaScorer(),
            ViolationLog(),
        )
        assert result.decision.name == "ALLOW"
