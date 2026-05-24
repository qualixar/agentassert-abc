# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for EU AI Act Report Generator."""

import json

from agentassert_abc.exporters.eu_ai_act import EUAIActReportGenerator, EUAIReport


class TestEUAIActReportGenerator:
    """EU AI Act Article 12 compliance report tests."""

    def test_record_and_generate(self) -> None:
        gen = EUAIActReportGenerator(system_name="Test System")
        gen.record_session(
            theta=0.93, c_bar=0.96, d_bar=0.04,
            violations=2, recoveries=2, recovery_success=2, turns=15,
        )
        gen.record_session(
            theta=0.91, c_bar=0.94, d_bar=0.06,
            violations=1, recoveries=1, recovery_success=0, turns=10,
        )

        report = gen.generate(
            contract_name="test-contract",
            contract_version="1.0.0",
            contract_summary="Test contract with 3 hard, 2 soft constraints.",
        )

        assert report.sessions_tracked == 2
        assert report.total_turns == 25
        assert report.total_violations == 3
        assert report.deployment_ready is True  # mean_theta = 0.92 >= 0.90

    def test_deployment_not_ready_below_threshold(self) -> None:
        gen = EUAIActReportGenerator()
        gen.record_session(
            theta=0.85, c_bar=0.80, d_bar=0.20,
            violations=5, recoveries=5, recovery_success=1, turns=10,
        )
        report = gen.generate()
        assert report.deployment_ready is False

    def test_article_15_needs_improvement(self) -> None:
        gen = EUAIActReportGenerator()
        gen.record_session(
            theta=0.85, c_bar=0.80, d_bar=0.20,
            violations=5, recoveries=5, recovery_success=1, turns=10,
        )
        report = gen.generate()
        assert report.article_15_accuracy == "needs_improvement"

    def test_article_14_review_required_low_recovery(self) -> None:
        gen = EUAIActReportGenerator()
        gen.record_session(
            theta=0.95, c_bar=0.98, d_bar=0.02,
            violations=10, recoveries=10, recovery_success=2, turns=50,
        )
        report = gen.generate()
        assert report.article_14_oversight == "review_required"

    def test_reset_clears_data(self) -> None:
        gen = EUAIActReportGenerator()
        gen.record_session(
            theta=0.90, c_bar=0.90, d_bar=0.10,
            violations=1, recoveries=0, recovery_success=0, turns=5,
        )
        gen.reset()
        report = gen.generate()
        assert report.sessions_tracked == 0

    def test_to_json_produces_valid_json(self) -> None:
        gen = EUAIActReportGenerator(system_name="json-test")
        gen.record_session(
            theta=0.92, c_bar=0.95, d_bar=0.05,
            violations=0, recoveries=0, recovery_success=0, turns=8,
        )
        report = gen.generate(
            contract_name="c", contract_version="1.0",
            contract_summary="Test.",
            recommendations=["Increase logging verbosity."],
        )
        raw = EUAIActReportGenerator.to_json(report)
        data = json.loads(raw)
        assert data["system"] == "json-test"
        assert data["contract"]["name"] == "c"
        assert len(data["recommendations"]) == 1

    def test_report_dataclass_defaults(self) -> None:
        r = EUAIReport()
        assert r.deployment_ready is False
        assert r.sessions_tracked == 0
