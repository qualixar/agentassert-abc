# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for SPRT Certification Engine — Patent §5.6.

Sequential Probability Ratio Test for agent compliance certification.
Tests written FIRST (TDD RED phase).
"""

from __future__ import annotations

import math

import pytest

from agentassert_abc.certification.sprt import (
    SPRTCertifier,
    SPRTDecision,
    SPRTResult,
    hoeffding_sample_size,
)

# ---------------------------------------------------------------------------
# hoeffding_sample_size — N_H = log(2/alpha) / (2 * epsilon^2)
# ---------------------------------------------------------------------------


class TestHoeffdingSampleSize:
    """Verify the Hoeffding bound formula from the patent."""

    def test_hoeffding_bound_basic(self) -> None:
        """N_H = ceil(log(2/alpha) / (2 * epsilon^2))."""
        alpha = 0.05
        epsilon = 0.05
        expected = math.ceil(math.log(2.0 / alpha) / (2.0 * epsilon**2))
        assert hoeffding_sample_size(alpha, epsilon) == expected

    def test_hoeffding_bound_tight_epsilon(self) -> None:
        """Tighter epsilon requires more samples."""
        n_loose = hoeffding_sample_size(0.05, 0.10)
        n_tight = hoeffding_sample_size(0.05, 0.05)
        assert n_tight > n_loose

    def test_hoeffding_bound_lower_alpha(self) -> None:
        """Lower alpha (higher confidence) requires more samples."""
        n_high_alpha = hoeffding_sample_size(0.10, 0.05)
        n_low_alpha = hoeffding_sample_size(0.01, 0.05)
        assert n_low_alpha > n_high_alpha

    def test_hoeffding_invalid_alpha(self) -> None:
        """Alpha must be in (0, 1)."""
        with pytest.raises(ValueError, match="alpha"):
            hoeffding_sample_size(0.0, 0.05)
        with pytest.raises(ValueError, match="alpha"):
            hoeffding_sample_size(1.0, 0.05)

    def test_hoeffding_invalid_epsilon(self) -> None:
        """Epsilon must be positive."""
        with pytest.raises(ValueError, match="epsilon"):
            hoeffding_sample_size(0.05, 0.0)
        with pytest.raises(ValueError, match="epsilon"):
            hoeffding_sample_size(0.05, -0.01)


# ---------------------------------------------------------------------------
# SPRTResult data class
# ---------------------------------------------------------------------------


class TestSPRTResult:
    """SPRTResult should carry decision, sessions_used, log_likelihood_ratio."""

    def test_result_fields(self) -> None:
        result = SPRTResult(
            decision=SPRTDecision.ACCEPT,
            sessions_used=42,
            log_likelihood_ratio=3.5,
        )
        assert result.decision == SPRTDecision.ACCEPT
        assert result.sessions_used == 42
        assert result.log_likelihood_ratio == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# SPRTCertifier — core sequential testing
# ---------------------------------------------------------------------------


class TestSPRTCertifierInit:
    """Constructor validation."""

    def test_valid_init(self) -> None:
        cert = SPRTCertifier(p0=0.90, p1=0.95, alpha=0.05, beta=0.10)
        assert cert.p0 == pytest.approx(0.90)
        assert cert.p1 == pytest.approx(0.95)

    def test_p0_must_be_less_than_p1(self) -> None:
        with pytest.raises(ValueError, match="p0.*p1"):
            SPRTCertifier(p0=0.95, p1=0.90, alpha=0.05, beta=0.10)

    def test_alpha_beta_range(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            SPRTCertifier(p0=0.90, p1=0.95, alpha=0.0, beta=0.10)
        with pytest.raises(ValueError, match="beta"):
            SPRTCertifier(p0=0.90, p1=0.95, alpha=0.05, beta=1.0)


class TestSPRTAcceptCompliantAgent:
    """Agent with high compliance (p=0.98) against threshold p0=0.90 -> ACCEPT."""

    def test_sprt_accept_compliant_agent(self) -> None:
        cert = SPRTCertifier(p0=0.90, p1=0.95, alpha=0.05, beta=0.10)
        # Simulate an agent that passes 98% of sessions
        # Feed 100 sessions: 98 pass, 2 fail (interleaved)
        result = None
        for i in range(200):
            passed = (i % 50) != 0  # fails on every 50th -> ~98%
            result = cert.update(session_passed=passed)
            if result.decision != SPRTDecision.CONTINUE:
                break

        assert result is not None
        assert result.decision == SPRTDecision.ACCEPT

    def test_sprt_fewer_sessions_than_fixed(self) -> None:
        """SPRT should use fewer sessions than Hoeffding for same confidence."""
        cert = SPRTCertifier(p0=0.90, p1=0.95, alpha=0.05, beta=0.10)
        epsilon = cert.p1 - cert.p0  # 0.05
        n_hoeffding = hoeffding_sample_size(0.05, epsilon)

        # Feed highly compliant agent (all pass)
        result = None
        for _i in range(n_hoeffding):
            result = cert.update(session_passed=True)
            if result.decision != SPRTDecision.CONTINUE:
                break

        assert result is not None
        assert result.decision == SPRTDecision.ACCEPT
        # SPRT should stop earlier — the 50-80% efficiency claim
        assert result.sessions_used < n_hoeffding


class TestSPRTRejectNonCompliant:
    """Agent with low compliance (p=0.80) against threshold p0=0.90 -> REJECT."""

    def test_sprt_reject_non_compliant(self) -> None:
        cert = SPRTCertifier(p0=0.90, p1=0.95, alpha=0.05, beta=0.10)
        # Simulate agent that passes 80% of sessions
        result = None
        for i in range(500):
            passed = (i % 5) != 0  # fails every 5th -> 80%
            result = cert.update(session_passed=passed)
            if result.decision != SPRTDecision.CONTINUE:
                break

        assert result is not None
        assert result.decision == SPRTDecision.REJECT


class TestSPRTStoppingTimeFormula:
    """E[N*] ~= log(1/alpha) / D_KL(p_hat || p0) from patent."""

    def test_stopping_time_formula(self) -> None:
        """Verify expected stopping time approximation is reasonable.

        For a strongly compliant agent (p_hat close to 1.0), the expected
        stopping time should be much less than Hoeffding's fixed bound.
        We verify the formula produces a positive finite number.
        """
        p0 = 0.90
        alpha = 0.05
        p_hat = 0.98  # observed compliance rate

        # KL divergence: D_KL(p_hat || p0) = p_hat*log(p_hat/p0) + (1-p_hat)*log((1-p_hat)/(1-p0))
        d_kl = p_hat * math.log(p_hat / p0) + (1 - p_hat) * math.log(
            (1 - p_hat) / (1 - p0)
        )
        expected_n = math.log(1.0 / alpha) / d_kl

        assert expected_n > 0
        assert math.isfinite(expected_n)
        # For p_hat=0.98 vs p0=0.90, expected stopping time should be
        # roughly in tens, not hundreds
        assert expected_n < 100


class TestSPRTIdempotency:
    """Calling update after a terminal decision should be safe."""

    def test_update_after_accept_returns_same(self) -> None:
        cert = SPRTCertifier(p0=0.50, p1=0.70, alpha=0.05, beta=0.10)
        # Feed enough passes to get ACCEPT
        result = None
        for _ in range(200):
            result = cert.update(session_passed=True)
            if result.decision != SPRTDecision.CONTINUE:
                break
        assert result is not None
        assert result.decision == SPRTDecision.ACCEPT

        # Further updates should return the same terminal decision
        extra = cert.update(session_passed=True)
        assert extra.decision == SPRTDecision.ACCEPT
