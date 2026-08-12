# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""OpenTelemetry Exporter — drift and compliance metrics as OTEL spans.

Exports each monitored turn as an OTEL span with attributes for:
- Compliance scores (hard, soft, total)
- Drift score D(t)
- Violation counts
- Theta (session-level span)

Designed as a subscriber to the EventBus for zero-overhead integration.
Gracefully degrades if opentelemetry-api is not installed (no-op mode).

Phase 6 — Layer 6: Dashboard & Export → OTEL.

Type C consolidation: `enforcement_request_span()`
and `enforcement_session_span()` were merged in from agentassert-typec's
`exporters/otel.py::TypeCOTelExporter` — additive span types on this same
exporter, not a second exporter class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OTelSpan:
    """Minimal OTEL span representation without importing the SDK.

    Contains enough data to be forwarded to any OTEL exporter or
    to be converted to a real span via the optional SDK integration.
    """

    name: str
    attributes: dict[str, Any]
    start_time_ns: int = 0
    end_time_ns: int = 0
    status: str = "OK"


class OTelExporter:
    """Export AgentAssert session data as OpenTelemetry spans.

    If otel_sdk_provider is None, operates in no-op mode (spans are
    collected but not sent anywhere). Callers can subclass and override
    _send() to integrate with their OTEL setup.

    Usage:
        exporter = OTelExporter(service_name="product-recommender")
        exporter.step_span(turn=3, c_total=0.95, drift=0.05,
                           hard_v=0, soft_v=0)
        exporter.session_span(theta=0.92, c_bar=0.96, ...)
    """

    def __init__(
        self,
        service_name: str = "agentassert",
        otel_sdk_provider: Any = None,
    ) -> None:
        self._service = service_name
        self._provider = otel_sdk_provider
        self._spans: list[OTelSpan] = []
        self._start_ns = time.monotonic_ns()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step_span(
        self,
        *,
        turn: int,
        c_total: float,
        drift: float,
        hard_violations: int,
        soft_violations: int,
        recovery_attempted: bool = False,
        recovery_succeeded: bool = False,
    ) -> OTelSpan:
        """Create a span for a single monitored turn."""
        span = OTelSpan(
            name=f"agentassert.step.{turn}",
            attributes={
                "service.name": self._service,
                "agentassert.turn": turn,
                "agentassert.compliance_total": c_total,
                "agentassert.drift_score": drift,
                "agentassert.hard_violations": hard_violations,
                "agentassert.soft_violations": soft_violations,
                "agentassert.recovery_attempted": recovery_attempted,
                "agentassert.recovery_succeeded": recovery_succeeded,
            },
            start_time_ns=self._start_ns,
            end_time_ns=time.monotonic_ns(),
            status="OK" if hard_violations == 0 else "ERROR",
        )
        self._spans.append(span)
        self._send(span)
        return span

    def session_span(
        self,
        *,
        theta: float,
        c_bar: float,
        d_bar: float,
        total_events: int,
        recovery_rate: float,
        turn_count: int,
    ) -> OTelSpan:
        """Create a session-level summary span."""
        span = OTelSpan(
            name="agentassert.session",
            attributes={
                "service.name": self._service,
                "agentassert.theta": theta,
                "agentassert.mean_compliance": c_bar,
                "agentassert.mean_drift": d_bar,
                "agentassert.total_events": total_events,
                "agentassert.recovery_rate": recovery_rate,
                "agentassert.turn_count": turn_count,
            },
            start_time_ns=self._start_ns,
            end_time_ns=time.monotonic_ns(),
            status="OK" if theta >= 0.9 else "WARN",
        )
        self._spans.append(span)
        self._send(span)
        return span

    def enforcement_request_span(
        self,
        *,
        session_id: str,
        request_id: str,
        contract_name: str,
        contract_version: str,
        decision: str,
        violation_name: str = "",
        violation_reason: str = "",
        tool: str = "",
        provider: str = "",
        overhead_ms: float = 0.0,
        stream: bool = False,
        adapter: str = "",
    ) -> OTelSpan:
        """Span for one enforcement-plane request (gateway/proxy/sdk/claude_code).

        Merged from agentassert-typec's `exporters/otel.py::TypeCOTelExporter`
        ( — additive span types on abc v2's existing exporter,
        not a second exporter). Unlike typec's original, this method does not
        require the real `opentelemetry-sdk` — it follows the same lightweight
        `OTelSpan` dataclass + `_send()` override point as `step_span`/
        `session_span`, so it degrades gracefully (no-op) with no extra deps.
        """
        span = OTelSpan(
            name="agentassert.enforcement.request",
            attributes={
                "service.name": self._service,
                "agentassert.request.id": request_id,
                "agentassert.session.id": session_id,
                "agentassert.contract.name": contract_name,
                "agentassert.contract.version": contract_version,
                "agentassert.decision": decision,
                "agentassert.violation.name": violation_name,
                "agentassert.violation.reason": violation_reason,
                "agentassert.tool": tool,
                "agentassert.provider": provider,
                "agentassert.overhead_ms": overhead_ms,
                "agentassert.stream": stream,
                "agentassert.adapter": adapter,
            },
            start_time_ns=self._start_ns,
            end_time_ns=time.monotonic_ns(),
            status="OK" if decision != "deny" else "ERROR",
        )
        self._spans.append(span)
        self._send(span)
        return span

    def enforcement_session_span(
        self,
        *,
        session_id: str,
        contract_name: str,
        theta: float,
        turn_count: int,
        violation_count: int,
        deny_count: int,
        duration_s: float,
        jsd: float,
        judge_cost_usd: float = 0.0,
        judge_samples: int = 0,
        judge_failures: int = 0,
    ) -> OTelSpan:
        """Session-level summary span for the enforcement plane.

        Merged from typec's `emit_session`. Distinct from
        `session_span` (the measurement plane's summary) — both may be
        emitted for the same session if a caller uses gateway enforcement
        alongside abc's measurement `SessionMonitor`.
        """
        span = OTelSpan(
            name="agentassert.enforcement.session",
            attributes={
                "service.name": self._service,
                "agentassert.session.id": session_id,
                "agentassert.contract.name": contract_name,
                "agentassert.theta": theta,
                "agentassert.turn_count": turn_count,
                "agentassert.violation_count": violation_count,
                "agentassert.deny_count": deny_count,
                "agentassert.duration_s": duration_s,
                "agentassert.drift.jsd": jsd,
                "agentassert.judge.cost_usd": judge_cost_usd,
                "agentassert.judge.samples": judge_samples,
                "agentassert.judge.failures": judge_failures,
            },
            start_time_ns=self._start_ns,
            end_time_ns=time.monotonic_ns(),
            status="OK" if theta >= 0.9 else "WARN",
        )
        self._spans.append(span)
        self._send(span)
        return span

    def flush(self) -> list[OTelSpan]:
        """Return and clear accumulated spans."""
        spans = list(self._spans)
        self._spans.clear()
        return spans

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send(self, span: OTelSpan) -> None:
        """Override point: forward span to a real OTEL exporter.

        Default implementation is no-op. Subclass or monkey-patch
        to integrate with opentelemetry-sdk.
        """
        pass
