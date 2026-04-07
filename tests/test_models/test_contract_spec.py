# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for ContractSpec Pydantic models — from patent §4.

Every model tested here maps directly to TECHNICAL-ATTACHMENT.md.
Models must be frozen (immutable), strictly typed, and match patent defaults.
"""

import pytest
from pydantic import ValidationError


class TestContractSpecMinimal:
    """Minimal valid contract creation."""

    def test_create_minimal_agent_contract(self) -> None:
        from agentassert_abc.models import ContractSpec

        spec = ContractSpec(
            contractspec="0.1",
            kind="agent",
            name="test-contract",
            description="A test contract",
            version="1.0.0",
        )
        assert spec.kind == "agent"
        assert spec.name == "test-contract"
        assert spec.contractspec == "0.1"

    def test_create_minimal_pipeline_contract(self) -> None:
        from agentassert_abc.models import ContractSpec

        spec = ContractSpec(
            contractspec="0.1",
            kind="pipeline",
            name="test-pipeline",
            description="A test pipeline contract",
            version="1.0.0",
        )
        assert spec.kind == "pipeline"

    def test_kind_must_be_agent_or_pipeline(self) -> None:
        from agentassert_abc.models import ContractSpec

        with pytest.raises(ValidationError):
            ContractSpec(
                contractspec="0.1",
                kind="invalid",  # type: ignore[arg-type]
                name="bad",
                description="bad",
                version="1.0.0",
            )

    def test_required_fields(self) -> None:
        from agentassert_abc.models import ContractSpec

        with pytest.raises(ValidationError):
            ContractSpec()  # type: ignore[call-arg]


class TestContractSpecFrozen:
    """Models must be immutable (LAW 13 + LAW 6)."""

    def test_contract_spec_is_frozen(self) -> None:
        from agentassert_abc.models import ContractSpec

        spec = ContractSpec(
            contractspec="0.1",
            kind="agent",
            name="test",
            description="test",
            version="1.0.0",
        )
        with pytest.raises(ValidationError):
            spec.name = "mutated"  # type: ignore[misc]


class TestConstraintCheck:
    """ConstraintCheck — the 12+2 operators from patent §4.3."""

    def test_equals_operator(self) -> None:
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="output.pii_detected", equals=False)
        assert check.field == "output.pii_detected"
        assert check.equals is False

    def test_gte_operator(self) -> None:
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="output.brand_tone_score", gte=0.7)
        assert check.gte == 0.7

    def test_lte_operator(self) -> None:
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="response.latency_ms", lte=3000)
        assert check.lte == 3000

    def test_in_operator(self) -> None:
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(
            field="output.category",
            in_=["electronics", "clothing"],
        )
        assert check.in_ == ["electronics", "clothing"]

    def test_contains_operator(self) -> None:
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="output.text", contains="sponsored")
        assert check.contains == "sponsored"

    def test_matches_operator(self) -> None:
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(
            field="output.discount_code",
            matches="^PROMO[0-9]{4}$",
        )
        assert check.matches == "^PROMO[0-9]{4}$"

    def test_exists_operator(self) -> None:
        from agentassert_abc.models import ConstraintCheck

        check = ConstraintCheck(field="context.prefs", exists=True)
        assert check.exists is True

    def test_field_is_required(self) -> None:
        from agentassert_abc.models import ConstraintCheck

        with pytest.raises(ValidationError):
            ConstraintCheck(equals=True)  # type: ignore[call-arg]


class TestHardSoftConstraints:
    """Hard/Soft constraint separation — patent §4.2."""

    def test_hard_constraint(self) -> None:
        from agentassert_abc.models import ConstraintCheck, HardConstraint

        c = HardConstraint(
            name="no-pii-leak",
            description="Never expose PII",
            category="confidentiality",
            check=ConstraintCheck(field="output.pii_detected", equals=False),
        )
        assert c.name == "no-pii-leak"
        assert c.check.equals is False

    def test_soft_constraint_with_recovery(self) -> None:
        from agentassert_abc.models import ConstraintCheck, SoftConstraint

        c = SoftConstraint(
            name="brand-tone",
            check=ConstraintCheck(field="output.brand_tone_score", gte=0.7),
            recovery="inject-tone-correction",
            recovery_window=2,
        )
        assert c.recovery == "inject-tone-correction"
        assert c.recovery_window == 2

    def test_soft_constraint_default_recovery_window(self) -> None:
        from agentassert_abc.models import ConstraintCheck, SoftConstraint

        c = SoftConstraint(
            name="test",
            check=ConstraintCheck(field="x", gte=0.5),
        )
        assert c.recovery_window == 3  # Patent default


class TestSatisfactionParams:
    """(p, delta, k)-satisfaction — patent §5.3."""

    def test_defaults_match_patent(self) -> None:
        """Defaults: p=0.95, delta=0.1, k=3 (from patent §5.3 e-commerce example)."""
        from agentassert_abc.models import SatisfactionParams

        params = SatisfactionParams()
        assert params.p == 0.95
        assert params.delta == 0.1
        assert params.k == 3

    def test_custom_values(self) -> None:
        from agentassert_abc.models import SatisfactionParams

        params = SatisfactionParams(p=0.99, delta=0.05, k=5)
        assert params.p == 0.99
        assert params.delta == 0.05
        assert params.k == 5


class TestDriftConfig:
    """Drift configuration — patent §5.1."""

    def test_default_weights_match_patent(self) -> None:
        """D(t) = 0.6 × D_compliance + 0.4 × D_distributional."""
        from agentassert_abc.models import DriftWeights

        w = DriftWeights()
        assert w.compliance == 0.6
        assert w.distributional == 0.4

    def test_weights_sum_to_one(self) -> None:
        from agentassert_abc.models import DriftWeights

        w = DriftWeights()
        assert abs(w.compliance + w.distributional - 1.0) < 0.01

    def test_default_thresholds_match_patent(self) -> None:
        from agentassert_abc.models import DriftThresholds

        t = DriftThresholds()
        assert t.warning == 0.3
        assert t.critical == 0.6

    def test_default_window(self) -> None:
        from agentassert_abc.models import DriftConfig

        d = DriftConfig()
        assert d.window == 50


class TestReliabilityConfig:
    """Reliability Index Θ — patent §5.7."""

    def test_default_weights_match_patent(self) -> None:
        """Θ = 0.35×C̄ + 0.25×(1-D̄) + 0.20×(1/(1+E)) + 0.20×S."""
        from agentassert_abc.models import ReliabilityWeights

        w = ReliabilityWeights()
        assert w.compliance == 0.35
        assert w.drift == 0.25
        assert w.event_freq == 0.20
        assert w.recovery_success == 0.20

    def test_weights_sum_to_one(self) -> None:
        from agentassert_abc.models import ReliabilityWeights

        w = ReliabilityWeights()
        total = w.compliance + w.drift + w.event_freq + w.recovery_success
        assert abs(total - 1.0) < 0.01

    def test_default_deployment_threshold(self) -> None:
        from agentassert_abc.models import ReliabilityConfig

        r = ReliabilityConfig()
        assert r.deployment_threshold == 0.90


class TestRecoveryAction:
    """Recovery strategies — patent §4.2 (4 types)."""

    def test_inject_correction(self) -> None:
        from agentassert_abc.models import RecoveryAction

        r = RecoveryAction(
            name="inject-tone-correction",
            type="inject_correction",
            actions=["Inject system message", "Re-score before delivery"],
            max_attempts=2,
            fallback="escalate-to-support",
        )
        assert r.type == "inject_correction"
        assert r.max_attempts == 2
        assert r.fallback == "escalate-to-support"

    def test_valid_recovery_types(self) -> None:
        from agentassert_abc.models import RecoveryAction

        valid_types = [
            "inject_correction",
            "reduce_autonomy",
            "pause_and_escalate",
            "graceful_shutdown",
        ]
        for rtype in valid_types:
            r = RecoveryAction(name="test", type=rtype)  # type: ignore[arg-type]
            assert r.type == rtype

    def test_invalid_recovery_type(self) -> None:
        from agentassert_abc.models import RecoveryAction

        with pytest.raises(ValidationError):
            RecoveryAction(name="test", type="invalid_type")  # type: ignore[arg-type]


class TestInvariants:
    """Invariants container — hard + soft separation."""

    def test_invariants_structure(self) -> None:
        from agentassert_abc.models import (
            ConstraintCheck,
            HardConstraint,
            Invariants,
            SoftConstraint,
        )

        inv = Invariants(
            hard=[
                HardConstraint(
                    name="no-pii",
                    check=ConstraintCheck(field="output.pii", equals=False),
                ),
            ],
            soft=[
                SoftConstraint(
                    name="tone",
                    check=ConstraintCheck(field="output.tone", gte=0.7),
                    recovery="fix-tone",
                ),
            ],
        )
        assert len(inv.hard) == 1
        assert len(inv.soft) == 1
