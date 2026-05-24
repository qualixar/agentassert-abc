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
