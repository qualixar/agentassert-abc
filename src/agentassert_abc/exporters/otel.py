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
