# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Enforcement policy for MCP tool calls."""

from __future__ import annotations

import pytest

from agentassert_abc.gateway.events import PostAction, PreAction
from agentassert_abc.mcp.guard import McpGuard

from .conftest import (
    StubEnforcer,
    allow,
    deny,
    modify,
    redact,
    tool_call,
    tool_result,
)


class TestPassthrough:
    """Anything that is not a tool call is relayed untouched."""

    @pytest.mark.parametrize(
        "message",
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {}},
            # A method no current spec revision defines. The guard must relay it
            # rather than fail, which is the point of not modelling the protocol.
            {"jsonrpc": "2.0", "id": 4, "method": "future/capability"},
        ],
    )
    def test_non_tool_call_forwarded_verbatim(self, message: dict) -> None:
        guard = McpGuard(StubEnforcer())
        relay = guard.on_client_message(message)
        assert relay.forward is message
        assert relay.reply is None

    def test_non_tool_traffic_is_never_scored(self) -> None:
        enforcer = StubEnforcer()
        guard = McpGuard(enforcer)
        guard.on_client_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert enforcer.events == []

    def test_tool_call_notification_is_relayed_not_denied(self) -> None:
        # No id means no reply is expected, so a DENY could not be delivered.
        guard = McpGuard(StubEnforcer([deny()]))
        relay = guard.on_client_message({"method": "tools/call", "params": {"name": "x"}})
        assert relay.forward is not None
        assert relay.reply is None


class TestDeny:
    """The load-bearing claim: a denied tool never reaches the server."""

    def test_denied_call_is_not_forwarded(self) -> None:
        guard = McpGuard(StubEnforcer([deny()]))
        relay = guard.on_client_message(tool_call(1, "rm"))
        assert relay.forward is None, "a denied tool call must never reach the server"

    def test_denied_call_answers_the_client_with_an_error_result(self) -> None:
        guard = McpGuard(StubEnforcer([deny(reason="destructive", violation="blocklist")]))
        relay = guard.on_client_message(tool_call(1, "rm"))
        assert relay.reply["id"] == 1
        assert relay.reply["result"]["isError"] is True
        text = relay.reply["result"]["content"][0]["text"]
        assert "destructive" in text
        assert "blocklist" in text
        assert "did not run" in text

    def test_denied_call_is_not_tracked_as_pending(self) -> None:
        guard = McpGuard(StubEnforcer([deny()]))
        guard.on_client_message(tool_call(1, "rm"))
        assert guard.pending_count == 0
        assert guard.deny_count == 1

    def test_deny_text_is_usable_when_the_contract_gave_no_reason(self) -> None:
        guard = McpGuard(StubEnforcer([deny(reason="", violation="")]))
        relay = guard.on_client_message(tool_call(1, "rm"))
        text = relay.reply["result"]["content"][0]["text"]
        assert "behavioral contract" in text
        assert "[]" not in text

    def test_real_contract_blocks_a_blocklisted_tool(self, safety_enforcer) -> None:
        guard = McpGuard(safety_enforcer)
        relay = guard.on_client_message(tool_call(1, "rm -rf /*"))
        assert relay.forward is None
        assert relay.reply["result"]["isError"] is True

    def test_real_contract_allows_an_unlisted_tool(self, safety_enforcer) -> None:
        guard = McpGuard(safety_enforcer)
        relay = guard.on_client_message(tool_call(1, "Read", {"path": "a.txt"}))
        assert relay.forward is not None
        assert relay.reply is None


class TestAllowAndModify:
    def test_allow_forwards_the_message_unchanged(self) -> None:
        guard = McpGuard(StubEnforcer([allow()]))
        msg = tool_call(1, "read", {"path": "a"})
        assert guard.on_client_message(msg).forward is msg

    def test_modify_rewrites_the_arguments(self) -> None:
        guard = McpGuard(StubEnforcer([modify({"path": "/tmp/safe"})]))
        relay = guard.on_client_message(tool_call(1, "read", {"path": "/etc/passwd"}))
        assert relay.forward["params"]["arguments"] == {"path": "/tmp/safe"}

    def test_modify_records_the_rewritten_args_for_the_response(self) -> None:
        # The PostAction must describe the call that actually ran, not the one
        # the agent asked for.
        enforcer = StubEnforcer([modify({"path": "/tmp/safe"}), allow()])
        guard = McpGuard(enforcer)
        guard.on_client_message(tool_call(1, "read", {"path": "/etc/passwd"}))
        guard.on_server_message(tool_result(1))
        post = next(e for e in enforcer.events if isinstance(e, PostAction))
        assert post.args == {"path": "/tmp/safe"}

    def test_modify_without_args_falls_back_to_forwarding_unchanged(self) -> None:
        from agentassert_abc.process.models import DecisionResult, TypeCDecision

        guard = McpGuard(
            StubEnforcer([DecisionResult(decision=TypeCDecision.MODIFY, modified_args=None)])
        )
        msg = tool_call(1, "read", {"path": "a"})
        assert guard.on_client_message(msg).forward is msg


class TestPreActionEvent:
    def test_pre_action_carries_the_tool_and_args(self) -> None:
        enforcer = StubEnforcer()
        guard = McpGuard(enforcer, session_id="s-1")
        guard.on_client_message(tool_call(1, "read_file", {"path": "/etc/passwd"}))
        event = enforcer.events[0]
        assert isinstance(event, PreAction)
        assert event.tool == "read_file"
        assert event.args == {"path": "/etc/passwd"}
        assert event.session_id == "s-1"
        assert event.contract_id == "stub"

    def test_session_id_is_generated_when_not_supplied(self) -> None:
        guard = McpGuard(StubEnforcer())
        assert guard.session_id.startswith("mcp-")


class TestResponseScoring:
    def test_response_state_uses_flat_dotted_output_keys(self) -> None:
        # The evaluator looks fields up as literal dotted keys (H-16). Shipping a
        # nested payload here is exactly the defect that scored compliant agents
        # as violating in 0.6.0 — assert the convention, not just that it ran.
        enforcer = StubEnforcer()
        guard = McpGuard(enforcer, server_label="files")
        guard.on_client_message(tool_call(1, "read"))
        guard.on_server_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": "hi"}],
                    "pii_detected": False,
                    "usage": {"tokens": 7},
                },
            }
        )
        post = next(e for e in enforcer.events if isinstance(e, PostAction))
        assert post.state["output.pii_detected"] is False
        assert post.state["output.usage.tokens"] == 7
        assert post.state["output.text"] == "hi"
        assert post.state["tool.name"] == "read"
        assert post.state["tool.server"] == "files"

    def test_real_governance_contract_can_read_its_own_invariant_field(
        self, governance_enforcer
    ) -> None:
        # full-governance.yaml declares `field: output.pii_detected`. If the
        # guard's flattening did not produce that key the invariant would score
        # False on every call regardless of behaviour.
        seen: list[object] = []
        original = governance_enforcer.evaluate
        governance_enforcer.evaluate = lambda e: (seen.append(e), original(e))[1]

        guard = McpGuard(governance_enforcer)
        guard.on_client_message(tool_call(1, "Read"))
        guard.on_server_message(tool_result(1, "hi", pii_detected=False))

        post = next(e for e in seen if isinstance(e, PostAction))
        assert "output.pii_detected" in post.state

    def test_response_to_an_untracked_id_is_relayed(self) -> None:
        guard = McpGuard(StubEnforcer())
        msg = tool_result(99)
        assert guard.on_server_message(msg).forward is msg

    def test_server_initiated_request_is_relayed(self) -> None:
        # Sampling/elicitation are the server asking the client for something,
        # not a tool executing — they carry no `result` and must pass through.
        guard = McpGuard(StubEnforcer())
        msg = {"jsonrpc": "2.0", "id": 5, "method": "sampling/createMessage", "params": {}}
        assert guard.on_server_message(msg).forward is msg

    def test_pending_call_is_released_after_its_response(self) -> None:
        guard = McpGuard(StubEnforcer())
        guard.on_client_message(tool_call(1, "read"))
        assert guard.pending_count == 1
        guard.on_server_message(tool_result(1))
        assert guard.pending_count == 0

    def test_interleaved_calls_resolve_to_the_right_pending_entry(self) -> None:
        # MCP is bidirectional and concurrent: responses may arrive out of order.
        enforcer = StubEnforcer()
        guard = McpGuard(enforcer)
        guard.on_client_message(tool_call(1, "alpha"))
        guard.on_client_message(tool_call(2, "beta"))
        guard.on_server_message(tool_result(2))
        guard.on_server_message(tool_result(1))
        tools = [e.tool for e in enforcer.events if isinstance(e, PostAction)]
        assert tools == ["beta", "alpha"]

    def test_string_and_int_ids_do_not_collide(self) -> None:
        enforcer = StubEnforcer()
        guard = McpGuard(enforcer)
        guard.on_client_message(tool_call(1, "int-call"))
        guard.on_client_message(tool_call("1", "str-call"))
        guard.on_server_message(tool_result("1"))
        post = next(e for e in enforcer.events if isinstance(e, PostAction))
        assert post.tool == "str-call"
        assert guard.pending_count == 1


class TestPostActionDeny:
    def test_output_is_withheld_when_the_response_violates(self) -> None:
        guard = McpGuard(StubEnforcer([allow(), deny(reason="leaked a key")]))
        guard.on_client_message(tool_call(1, "read"))
        relay = guard.on_server_message(tool_result(1, "sk-secret"))
        assert "sk-secret" not in str(relay.forward)
        assert relay.forward["result"]["isError"] is True

    def test_withheld_text_says_the_tool_did_run(self) -> None:
        # A PostAction DENY cannot un-execute the call. Reporting it like a
        # PreAction DENY would misstate what the guard actually prevented.
        guard = McpGuard(StubEnforcer([allow(), deny(reason="leaked")]))
        guard.on_client_message(tool_call(1, "read"))
        relay = guard.on_server_message(tool_result(1, "secret"))
        text = relay.forward["result"]["content"][0]["text"]
        assert "executed" in text
        assert "withheld" in text

    def test_pii_filter_deny_withholds_the_output(self, monkeypatch) -> None:
        # The contract's own decision allows the response; the PII filter is a
        # second, independent gate over the returned text.
        import agentassert_abc.enforce.bridge as guard_mod

        monkeypatch.setattr(
            guard_mod,
            "evaluate_pii_filter",
            lambda *_a, **_k: deny(reason="ssn detected", violation="pii_filter"),
        )
        guard = McpGuard(StubEnforcer([allow(), allow()]))
        guard.on_client_message(tool_call(1, "read"))
        relay = guard.on_server_message(tool_result(1, "123-45-6789"))
        assert "123-45-6789" not in str(relay.forward)
        assert relay.forward["result"]["isError"] is True
        assert "ssn detected" in relay.forward["result"]["content"][0]["text"]
        assert guard.deny_count == 1

    def test_pii_filter_redact_masks_without_withholding(self, monkeypatch) -> None:
        import agentassert_abc.enforce.bridge as guard_mod

        monkeypatch.setattr(guard_mod, "evaluate_pii_filter", lambda *_a, **_k: redact())
        monkeypatch.setattr(guard_mod, "apply_pii_redaction", lambda _t, _p: "[REDACTED:SSN]")
        guard = McpGuard(StubEnforcer([allow(), allow()]))
        guard.on_client_message(tool_call(1, "read"))
        relay = guard.on_server_message(tool_result(1, "123-45-6789"))
        assert relay.forward["result"]["content"][0]["text"] == "[REDACTED:SSN]"
        assert relay.forward["result"].get("isError") is not True

    def test_empty_result_text_skips_the_pii_filter(self, monkeypatch) -> None:
        called: list[int] = []
        import agentassert_abc.enforce.bridge as guard_mod

        monkeypatch.setattr(guard_mod, "evaluate_pii_filter", lambda *_a, **_k: called.append(1))
        guard = McpGuard(StubEnforcer([allow(), allow()]))
        guard.on_client_message(tool_call(1, "read"))
        guard.on_server_message({"jsonrpc": "2.0", "id": 1, "result": {"content": []}})
        assert called == []

    def test_redact_on_the_response_masks_matched_patterns(self, monkeypatch) -> None:
        import agentassert_abc.enforce.bridge as guard_mod

        monkeypatch.setattr(guard_mod, "apply_pii_redaction", lambda text, _p: "[MASKED]")
        guard = McpGuard(StubEnforcer([allow(), redact()]))
        guard.on_client_message(tool_call(1, "read"))
        relay = guard.on_server_message(tool_result(1, "secret"))
        assert relay.forward["result"]["content"][0]["text"] == "[MASKED]"

    def test_redact_decided_at_pre_action_still_applies_to_the_response(self, monkeypatch) -> None:
        import agentassert_abc.enforce.bridge as guard_mod

        monkeypatch.setattr(guard_mod, "apply_pii_redaction", lambda text, _p: "[MASKED]")
        guard = McpGuard(StubEnforcer([redact(), allow()]))
        guard.on_client_message(tool_call(1, "read"))
        relay = guard.on_server_message(tool_result(1, "secret"))
        assert relay.forward["result"]["content"][0]["text"] == "[MASKED]"


class TestFailureModes:
    def test_fail_open_relays_the_original_request(self) -> None:
        # The bug this pins: returning `forward=None` here would silently DROP
        # the call, hanging the client on a response that never comes.
        guard = McpGuard(StubEnforcer(raises=RuntimeError("boom")))
        msg = tool_call(1, "read")
        relay = guard.on_client_message(msg)
        assert relay.forward is msg
        assert relay.reply is None

    def test_fail_open_does_not_track_the_call(self) -> None:
        # Evaluation is already broken for it; scoring its response would report
        # a violation caused by the guard's own fault.
        guard = McpGuard(StubEnforcer(raises=RuntimeError("boom")))
        guard.on_client_message(tool_call(1, "read"))
        assert guard.pending_count == 0

    def test_fail_closed_denies_what_it_cannot_evaluate(self) -> None:
        guard = McpGuard(StubEnforcer(raises=RuntimeError("boom")), fail_closed=True)
        relay = guard.on_client_message(tool_call(1, "read"))
        assert relay.forward is None
        assert relay.reply["result"]["isError"] is True
        assert "fail-closed" in relay.reply["result"]["content"][0]["text"]

    def test_response_scoring_failure_still_returns_the_response(self) -> None:
        # Deliberately fail-open even under fail_closed: the tool already ran, so
        # withholding output punishes the agent for the guard's bug without
        # preventing any side effect.
        enforcer = StubEnforcer([allow()])
        guard = McpGuard(enforcer, fail_closed=True)
        guard.on_client_message(tool_call(1, "read"))
        enforcer._raises = RuntimeError("boom")
        msg = tool_result(1, "payload")
        assert guard.on_server_message(msg).forward is msg
