# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for OpenTelemetry Exporter."""

from agentassert_abc.exporters.otel import OTelExporter, OTelSpan


class TestOTelExporter:
    """OTEL span export tests."""

    def test_step_span_creates_span(self) -> None:
        exporter = OTelExporter(service_name="test-svc")
        span = exporter.step_span(
            turn=3, c_total=0.95, drift=0.05,
            hard_violations=0, soft_violations=0,
        )
        assert span.name == "agentassert.step.3"
        assert span.attributes["agentassert.turn"] == 3
        assert span.attributes["agentassert.compliance_total"] == 0.95
        assert span.attributes["agentassert.drift_score"] == 0.05
        assert span.status == "OK"

    def test_step_span_status_error_on_hard_violation(self) -> None:
        exporter = OTelExporter()
        span = exporter.step_span(
            turn=1, c_total=0.7, drift=0.3,
            hard_violations=1, soft_violations=0,
        )
        assert span.status == "ERROR"

    def test_session_span_creates_span(self) -> None:
        exporter = OTelExporter(service_name="agent")
        span = exporter.session_span(
            theta=0.92, c_bar=0.96, d_bar=0.04,
            total_events=2, recovery_rate=0.5, turn_count=10,
        )
        assert span.name == "agentassert.session"
        assert span.attributes["agentassert.theta"] == 0.92
        assert span.attributes["agentassert.turn_count"] == 10

    def test_session_span_warn_below_threshold(self) -> None:
        exporter = OTelExporter()
        span = exporter.session_span(
            theta=0.85, c_bar=0.80, d_bar=0.20,
            total_events=5, recovery_rate=0.3, turn_count=10,
        )
        assert span.status == "WARN"

    def test_flush_returns_and_clears(self) -> None:
        exporter = OTelExporter()
        exporter.step_span(
            turn=1, c_total=1.0, drift=0.0,
            hard_violations=0, soft_violations=0,
        )
        exporter.step_span(
            turn=2, c_total=0.9, drift=0.1,
            hard_violations=0, soft_violations=1,
        )
        spans = exporter.flush()
        assert len(spans) == 2
        assert len(exporter.flush()) == 0  # cleared

    def test_noop_by_default(self) -> None:
        exporter = OTelExporter()
        span = exporter.step_span(
            turn=1, c_total=1.0, drift=0.0,
            hard_violations=0, soft_violations=0,
        )
        assert isinstance(span, OTelSpan)


class TestEnforcementSpans:
    """Type C consolidation (the v1 to v2 migration) — merged enforcement-plane
    span types from agentassert-typec's `TypeCOTelExporter`, additive on this
    same `OTelExporter` (not a second exporter class).
    """

    def test_enforcement_request_span_allow(self) -> None:
        exporter = OTelExporter(service_name="test-svc")
        span = exporter.enforcement_request_span(
            session_id="s1",
            request_id="r1",
            contract_name="safety-minimal",
            contract_version="0.1",
            decision="allow",
            tool="Read",
            provider="anthropic",
            overhead_ms=1.2,
        )
        assert span.name == "agentassert.enforcement.request"
        assert span.attributes["agentassert.decision"] == "allow"
        assert span.attributes["agentassert.tool"] == "Read"
        assert span.status == "OK"

    def test_enforcement_request_span_deny_is_error(self) -> None:
        exporter = OTelExporter()
        span = exporter.enforcement_request_span(
            session_id="s1",
            request_id="r1",
            contract_name="safety-minimal",
            contract_version="0.1",
            decision="deny",
            violation_name="tool_blocklist",
            violation_reason="blocked tool",
            tool="bash",
        )
        assert span.status == "ERROR"
        assert span.attributes["agentassert.violation.name"] == "tool_blocklist"

    def test_enforcement_session_span_ok_above_threshold(self) -> None:
        exporter = OTelExporter(service_name="agent")
        span = exporter.enforcement_session_span(
            session_id="s1",
            contract_name="partner-mode",
            theta=0.95,
            turn_count=10,
            violation_count=0,
            deny_count=0,
            duration_s=12.5,
            jsd=0.02,
        )
        assert span.name == "agentassert.enforcement.session"
        assert span.status == "OK"
        assert span.attributes["agentassert.theta"] == 0.95

    def test_enforcement_session_span_warn_below_threshold(self) -> None:
        exporter = OTelExporter()
        span = exporter.enforcement_session_span(
            session_id="s1",
            contract_name="partner-mode",
            theta=0.5,
            turn_count=10,
            violation_count=3,
            deny_count=1,
            duration_s=12.5,
            jsd=0.4,
            judge_cost_usd=0.01,
            judge_samples=2,
            judge_failures=1,
        )
        assert span.status == "WARN"
        assert span.attributes["agentassert.judge.failures"] == 1

    def test_enforcement_spans_flush_alongside_measurement_spans(self) -> None:
        exporter = OTelExporter()
        exporter.step_span(turn=1, c_total=1.0, drift=0.0, hard_violations=0, soft_violations=0)
        exporter.enforcement_request_span(
            session_id="s1",
            request_id="r1",
            contract_name="c",
            contract_version="0.1",
            decision="allow",
        )
        spans = exporter.flush()
        assert len(spans) == 2
        assert len(exporter.flush()) == 0
