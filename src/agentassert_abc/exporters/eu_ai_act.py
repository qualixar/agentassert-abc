# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""EU AI Act Article 12 Compliance Report Generator.

Generates structured compliance evidence from AgentAssert session data
for EU AI Act Article 12 (Record-Keeping), Article 14 (Human Oversight),
and Article 15 (Accuracy, Robustness, Cybersecurity).

The report is a machine-readable JSON document that serves as
auditable evidence of behavioral contract enforcement during
high-risk AI system operation.

Phase 6 — Layer 6: Dashboard & Export → EU AI Act Report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class EUAIReport:
    """Immutable EU AI Act compliance report.

    Attributes:
        report_id: Unique identifier for this report.
        generated_at: ISO 8601 timestamp.
        system_name: Name of the monitored AI system.
        contract_name: ContractSpec name.
        contract_version: ContractSpec version.
        article_12_logging: Whether automatic logging is in place.
        article_14_oversight: Human oversight verification status.
        article_15_accuracy: Accuracy/robustness assessment.
        sessions_tracked: Number of sessions monitored.
        total_turns: Total interaction turns across all sessions.
        mean_theta: Mean Reliability Index across sessions.
        mean_compliance: Mean compliance score (hard+soft avg).
        mean_drift: Mean drift score.
        total_violations: Total violation events observed.
        recovery_rate: Fraction of recoveries that succeeded.
        deployment_ready: Whether Theta >= 0.90 threshold.
        contract_summary: Human-readable contract constraint summary.
        recommendations: Optional compliance recommendations.
    """

    report_id: str = ""
    generated_at: str = ""
    system_name: str = ""
    contract_name: str = ""
    contract_version: str = ""

    article_12_logging: bool = True
    article_14_oversight: str = "verified"
    article_15_accuracy: str = "assessed"

    sessions_tracked: int = 0
    total_turns: int = 0
    mean_theta: float = 0.0
    mean_compliance: float = 0.0
    mean_drift: float = 0.0
    total_violations: int = 0
    recovery_rate: float = 0.0
    deployment_ready: bool = False

    contract_summary: str = ""
    recommendations: list[str] = field(default_factory=list)


class EUAIActReportGenerator:
    """Generates EU AI Act Article 12 compliance reports from session data.

    Usage:
        gen = EUAIActReportGenerator(system_name="Product Rec Engine")
        gen.record_session(theta=0.92, c_bar=0.96, d_bar=0.04,
                           violations=2, recoveries=1, recovery_success=1,
                           turns=12)
        report = gen.generate(contract_name="ecommerce-rec", version="2.1.0")
        print(report.to_json())
    """

    DEPLOYMENT_THETA_THRESHOLD = 0.90

    def __init__(self, system_name: str = "unknown") -> None:
        self._system = system_name
        self._sessions: int = 0
        self._total_turns: int = 0
        self._theta_sum: float = 0.0
        self._compliance_sum: float = 0.0
        self._drift_sum: float = 0.0
        self._violations: int = 0
        self._recovery_attempts: int = 0
        self._recovery_successes: int = 0

    def record_session(
        self,
        *,
        theta: float,
        c_bar: float,
        d_bar: float,
        violations: int,
        recoveries: int,
        recovery_success: int,
        turns: int,
    ) -> None:
        """Record one monitored session's aggregate data."""
        self._sessions += 1
        self._total_turns += turns
        self._theta_sum += theta
        self._compliance_sum += c_bar
        self._drift_sum += d_bar
        self._violations += violations
        self._recovery_attempts += recoveries
        self._recovery_successes += recovery_success

    def generate(
        self,
        contract_name: str = "",
        contract_version: str = "",
        contract_summary: str = "",
        recommendations: list[str] | None = None,
    ) -> EUAIReport:
        """Build the compliance report from recorded sessions."""
        n = max(self._sessions, 1)
        mean_theta = self._theta_sum / n
        recovery_rate = (
            self._recovery_successes / max(self._recovery_attempts, 1)
        )

        return EUAIReport(
            report_id=f"EUAI-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
            generated_at=datetime.now(UTC).isoformat(),
            system_name=self._system,
            contract_name=contract_name,
            contract_version=contract_version,
            article_12_logging=True,
            article_14_oversight=(
                "verified" if recovery_rate >= 0.5 else "review_required"
            ),
            article_15_accuracy=(
                "assessed" if mean_theta >= self.DEPLOYMENT_THETA_THRESHOLD
                else "needs_improvement"
            ),
            sessions_tracked=self._sessions,
            total_turns=self._total_turns,
            mean_theta=round(mean_theta, 4),
            mean_compliance=round(self._compliance_sum / n, 4),
            mean_drift=round(self._drift_sum / n, 4),
            total_violations=self._violations,
            recovery_rate=round(recovery_rate, 4),
            deployment_ready=mean_theta >= self.DEPLOYMENT_THETA_THRESHOLD,
            contract_summary=contract_summary,
            recommendations=recommendations or [],
        )

    def reset(self) -> None:
        """Clear all recorded session data."""
        self._sessions = 0
        self._total_turns = 0
        self._theta_sum = 0.0
        self._compliance_sum = 0.0
        self._drift_sum = 0.0
        self._violations = 0
        self._recovery_attempts = 0
        self._recovery_successes = 0

    @staticmethod
    def to_json(report: EUAIReport, indent: int = 2) -> str:
        """Serialize report to indented JSON string."""
        return json.dumps(
            {
                "report_id": report.report_id,
                "generated_at": report.generated_at,
                "system": report.system_name,
                "contract": {
                    "name": report.contract_name,
                    "version": report.contract_version,
                },
                "compliance": {
                    "article_12_logging": report.article_12_logging,
                    "article_14_human_oversight": report.article_14_oversight,
                    "article_15_accuracy_robustness": report.article_15_accuracy,
                },
                "metrics": {
                    "sessions_tracked": report.sessions_tracked,
                    "total_turns": report.total_turns,
                    "mean_theta": report.mean_theta,
                    "mean_compliance": report.mean_compliance,
                    "mean_drift": report.mean_drift,
                    "total_violations": report.total_violations,
                    "recovery_rate": report.recovery_rate,
                    "deployment_ready": report.deployment_ready,
                },
                "contract_summary": report.contract_summary,
                "recommendations": report.recommendations,
            },
            indent=indent,
        )
