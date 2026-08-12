# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""The framework-neutral enforcement bridge."""

from __future__ import annotations

import pytest

from agentassert_abc.enforce import EnforcementBridge, bridge_from_yaml
from agentassert_abc.exceptions import ContractBreachError, ContractLoadError
from agentassert_abc.gateway.events import PostAction, PreAction

from ..test_mcp.conftest import CONTRACTS, StubEnforcer, allow, deny, modify, redact


def bridge(decisions=None, **kwargs) -> EnforcementBridge:
    return EnforcementBridge(StubEnforcer(decisions), **kwargs)


class TestBeforeTool:
    def test_allow_passes_the_arguments_through(self) -> None:
        decision = bridge([allow()]).before_tool("read", {"path": "a"})
        assert decision.allowed is True
        assert decision.arguments == {"path": "a"}
        assert decision.modified is False

    def test_deny_reports_reason_and_violation(self) -> None:
        decision = bridge([deny(reason="destructive", violation="blocklist")]).before_tool("rm")
        assert decision.allowed is False
        assert decision.reason == "destructive"
        assert decision.violation == "blocklist"

    def test_deny_supplies_a_reason_when_the_contract_gave_none(self) -> None:
        # A shim surfaces `reason` to the model; an empty string would tell the
        # agent nothing about how to proceed.
        decision = bridge([deny(reason="", violation="")]).before_tool("rm")
        assert "behavioral contract" in decision.reason

    def test_modify_returns_the_rewritten_arguments(self) -> None:
        decision = bridge([modify({"path": "/tmp/safe"})]).before_tool("read", {"path": "/etc"})
        assert decision.allowed is True
        assert decision.modified is True
        assert decision.arguments == {"path": "/tmp/safe"}

    def test_arguments_are_copied_not_aliased(self) -> None:
        # A shim hands these straight to the framework; sharing the dict would
        # let a later mutation rewrite what the audit log recorded.
        args = {"path": "a"}
        decision = bridge([allow()]).before_tool("read", args)
        decision.arguments["path"] = "b"
        assert args == {"path": "a"}

    def test_pre_action_event_carries_tool_and_args(self) -> None:
        enforcer = StubEnforcer()
        EnforcementBridge(enforcer, session_id="s1").before_tool("read", {"p": 1})
        event = enforcer.events[0]
        assert isinstance(event, PreAction)
        assert (event.tool, event.args, event.session_id) == ("read", {"p": 1}, "s1")

    def test_none_args_is_treated_as_empty(self) -> None:
        assert bridge([allow()]).before_tool("read").arguments == {}


class TestVerdictErgonomics:
    def test_decision_is_truthy_when_allowed(self) -> None:
        # Lets a shim write `return bool(decision)` for CrewAI's convention.
        assert bool(bridge([allow()]).before_tool("read")) is True
        assert bool(bridge([deny()]).before_tool("rm")) is False

    def test_raise_if_denied_is_silent_when_allowed(self) -> None:
        bridge([allow()]).before_tool("read").raise_if_denied()

    def test_raise_if_denied_raises_contract_breach(self) -> None:
        decision = bridge([deny(reason="nope", violation="v")]).before_tool("rm")
        with pytest.raises(ContractBreachError) as exc:
            decision.raise_if_denied()
        assert "nope" in str(exc.value)

    def test_outcome_is_truthy_and_raises_the_same_way(self) -> None:
        b = bridge([allow(), deny(reason="leak")])
        b.before_tool("read")
        outcome = b.after_tool("read", {}, {"text": "secret"})
        assert bool(outcome) is False
        with pytest.raises(ContractBreachError):
            outcome.raise_if_denied()


class TestAfterTool:
    def test_allowed_result_is_returned_unchanged(self) -> None:
        outcome = bridge([allow()]).after_tool("read", {}, {"a": 1})
        assert outcome.allowed is True
        assert outcome.result == {"a": 1}
        assert outcome.redacted is False

    def test_denied_result_is_flagged_for_withholding(self) -> None:
        outcome = bridge([deny(reason="leaked")]).after_tool("read", {}, {"a": 1})
        assert outcome.allowed is False
        assert outcome.reason == "leaked"

    def test_state_uses_flat_dotted_output_keys(self) -> None:
        # H-16: the evaluator reads literal dotted keys. This is the convention
        # whose absence made compliant agents score as violating in 0.6.0.
        enforcer = StubEnforcer()
        EnforcementBridge(enforcer).after_tool("read", {}, {"pii_detected": False, "u": {"t": 7}})
        post = next(e for e in enforcer.events if isinstance(e, PostAction))
        assert post.state["output.pii_detected"] is False
        assert post.state["output.u.t"] == 7
        assert post.state["tool.name"] == "read"

    def test_base_state_is_merged_into_every_result(self) -> None:
        enforcer = StubEnforcer()
        EnforcementBridge(enforcer, base_state={"tool.server": "github"}).after_tool("read")
        post = next(e for e in enforcer.events if isinstance(e, PostAction))
        assert post.state["tool.server"] == "github"

    def test_explicit_text_wins_over_flattening(self) -> None:
        enforcer = StubEnforcer()
        EnforcementBridge(enforcer).after_tool("read", {}, {"a": 1}, text="the real text")
        post = next(e for e in enforcer.events if isinstance(e, PostAction))
        assert post.state["output.text"] == "the real text"

    def test_recorded_args_describe_the_call_that_ran(self) -> None:
        enforcer = StubEnforcer([modify({"path": "/tmp/safe"}), allow()])
        b = EnforcementBridge(enforcer)
        decision = b.before_tool("read", {"path": "/etc/passwd"})
        b.after_tool("read", decision.arguments, {})
        post = next(e for e in enforcer.events if isinstance(e, PostAction))
        assert post.args == {"path": "/tmp/safe"}


class TestRedaction:
    def test_redact_verdict_masks_the_text(self, monkeypatch) -> None:
        import agentassert_abc.enforce.bridge as mod

        monkeypatch.setattr(mod, "apply_pii_redaction", lambda _t, _p: "[MASKED]")
        outcome = bridge([redact()]).after_tool("read", {}, {"text": "secret"}, text="secret")
        assert outcome.allowed is True
        assert outcome.redacted is True
        assert outcome.redacted_text == "[MASKED]"

    def test_force_redact_applies_a_decision_made_before_the_output_existed(
        self, monkeypatch
    ) -> None:
        import agentassert_abc.enforce.bridge as mod

        monkeypatch.setattr(mod, "apply_pii_redaction", lambda _t, _p: "[MASKED]")
        b = bridge([redact(), allow()])
        decision = b.before_tool("read")
        assert decision.redact_result is True
        outcome = b.after_tool("read", {}, {}, text="secret", force_redact=decision.redact_result)
        assert outcome.redacted_text == "[MASKED]"

    def test_pii_filter_deny_withholds_the_output(self, monkeypatch) -> None:
        import agentassert_abc.enforce.bridge as mod

        monkeypatch.setattr(
            mod, "evaluate_pii_filter", lambda *_a, **_k: deny(reason="ssn", violation="pii")
        )
        outcome = bridge([allow()]).after_tool("read", {}, {}, text="123-45-6789")
        assert outcome.allowed is False
        assert outcome.reason == "ssn"

    def test_no_text_means_no_redaction_attempt(self) -> None:
        outcome = bridge([redact()]).after_tool("read", {}, None)
        assert outcome.redacted is False


class TestFailureModes:
    def test_fail_open_allows_and_marks_the_call_unevaluated(self) -> None:
        # `evaluated` is what lets a caller skip the paired post-hook rather
        # than score a result against a contract that already failed to run.
        decision = EnforcementBridge(StubEnforcer(raises=RuntimeError("boom"))).before_tool(
            "read", {"a": 1}
        )
        assert decision.allowed is True
        assert decision.evaluated is False
        assert decision.arguments == {"a": 1}

    def test_fail_closed_denies_and_marks_it_unevaluated(self) -> None:
        decision = EnforcementBridge(
            StubEnforcer(raises=RuntimeError("boom")), fail_closed=True, surface="crewai"
        ).before_tool("read")
        assert decision.allowed is False
        assert decision.evaluated is False
        assert "fail-closed" in decision.reason
        assert "crewai" in decision.reason

    def test_a_successful_call_is_marked_evaluated(self) -> None:
        assert bridge([allow()]).before_tool("read").evaluated is True

    def test_result_scoring_failure_always_fails_open(self) -> None:
        # Even under fail_closed: the tool already ran, so withholding output
        # punishes the agent for the bridge's bug without preventing anything.
        b = EnforcementBridge(StubEnforcer(raises=RuntimeError("boom")), fail_closed=True)
        outcome = b.after_tool("read", {}, {"a": 1})
        assert outcome.allowed is True
        assert outcome.result == {"a": 1}


class TestTurnAndSession:
    def test_turn_hooks_do_not_raise_when_the_enforcer_fails(self) -> None:
        b = EnforcementBridge(StubEnforcer(raises=RuntimeError("boom")))
        b.start_turn("hello")
        b.end_turn("goodbye")

    def test_close_is_safe_when_the_enforcer_fails(self) -> None:
        assert EnforcementBridge(StubEnforcer(raises=RuntimeError("x"))).close() is None

    def test_deny_count_accumulates_across_both_boundaries(self) -> None:
        b = bridge([deny(), allow(), deny()])
        b.before_tool("a")
        b.before_tool("b")
        b.after_tool("b", {}, {})
        assert b.deny_count == 2

    def test_identity_is_exposed_for_logging(self) -> None:
        b = EnforcementBridge(StubEnforcer(), surface="crewai", session_id="s9")
        assert (b.surface, b.session_id, b.contract_id) == ("crewai", "s9", "stub")

    def test_underlying_enforcer_is_reachable_for_session_metrics(self) -> None:
        enforcer = StubEnforcer()
        assert EnforcementBridge(enforcer).enforcer is enforcer

    def test_session_id_is_generated_from_the_surface(self) -> None:
        assert EnforcementBridge(StubEnforcer(), surface="langchain").session_id.startswith(
            "langchain-"
        )


class TestBridgeFromYaml:
    def test_loads_a_real_contract(self) -> None:
        b = bridge_from_yaml(str(CONTRACTS / "safety-minimal.yaml"), surface="test")
        assert b.contract_id == "safety-minimal"

    def test_real_contract_denies_a_blocklisted_tool(self) -> None:
        b = bridge_from_yaml(str(CONTRACTS / "safety-minimal.yaml"))
        assert b.before_tool("rm -rf /*").allowed is False
        assert b.before_tool("Read", {"path": "a"}).allowed is True

    def test_missing_contract_raises(self) -> None:
        with pytest.raises(ContractLoadError):
            bridge_from_yaml("/nonexistent/contract.yaml")
