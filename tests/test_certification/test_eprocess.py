# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for GraphEProcess e-process certification — LLD-C Thm C.1–C.3.

Written FIRST (TDD RED phase). All tests imported from a module that does not
yet exist, so pytest will report collection errors until eprocess.py is
implemented.

Mathematical references throughout: LLD-C v2 §3 (Thm C.1), §4 (Thm C.2),
§5 (Thm C.3), §14 (test plan).
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from agentassert_abc.certification.eprocess import (
    EProcessError,
    EProcessUpdate,
    GraphEProcess,
    kl_bernoulli,
    simulate_type1_crossing_rate,
)

# ── Configuration Validation ───────────────────────────────────────────────


class TestConfigValidation:
    """Invalid construction parameters raise EProcessError — LLD-C §8.1."""

    def test_p0_zero_raises(self) -> None:
        with pytest.raises(EProcessError, match="p0"):
            GraphEProcess.fixed_lambda(p0=0.0, alpha=0.05, lam=0.5)

    def test_p0_one_raises(self) -> None:
        with pytest.raises(EProcessError, match="p0"):
            GraphEProcess.fixed_lambda(p0=1.0, alpha=0.05, lam=0.5)

    def test_alpha_zero_raises(self) -> None:
        with pytest.raises(EProcessError, match="alpha"):
            GraphEProcess.fixed_lambda(p0=0.8, alpha=0.0, lam=0.5)

    def test_alpha_one_raises(self) -> None:
        with pytest.raises(EProcessError, match="alpha"):
            GraphEProcess.fixed_lambda(p0=0.8, alpha=1.0, lam=0.5)

    def test_epsilon_zero_raises(self) -> None:
        with pytest.raises(EProcessError, match="epsilon"):
            GraphEProcess.mixture(p0=0.8, alpha=0.05, epsilon=0.0)

    def test_epsilon_at_one_minus_p0_raises(self) -> None:
        """epsilon must satisfy 0 < epsilon < 1-p0 — LLD-C §5.1."""
        with pytest.raises(EProcessError, match="epsilon"):
            GraphEProcess.mixture(p0=0.8, alpha=0.05, epsilon=0.2)  # >= 1-0.8=0.2

    def test_epsilon_above_one_minus_p0_raises(self) -> None:
        with pytest.raises(EProcessError, match="epsilon"):
            GraphEProcess.mixture(p0=0.6, alpha=0.05, epsilon=0.5)

    def test_sprt_p1_not_greater_than_p0_raises(self) -> None:
        with pytest.raises(EProcessError):
            GraphEProcess.from_sprt(p0=0.8, alpha=0.05, p1=0.7)

    def test_sprt_p1_equals_p0_raises(self) -> None:
        with pytest.raises(EProcessError):
            GraphEProcess.from_sprt(p0=0.8, alpha=0.05, p1=0.8)


# ── Lambda Range Enforcement ───────────────────────────────────────────────


class TestLambdaRange:
    """Betting fraction must be in [0, 1/p0] — LLD-C Eq (3.1), Thm C.1."""

    def test_negative_lambda_raises(self) -> None:
        """Negative lambda violates the one-sided null. LLD-C §3.1."""
        with pytest.raises(EProcessError, match="lambda"):
            GraphEProcess.fixed_lambda(p0=0.8, alpha=0.05, lam=-0.01)

    def test_lambda_strictly_above_upper_bound_raises(self) -> None:
        """lambda > 1/p0 makes failure factor negative. LLD-C §14 test #3."""
        p0 = 0.8
        with pytest.raises(EProcessError, match="lambda"):
            GraphEProcess.fixed_lambda(p0=p0, alpha=0.05, lam=1.0 / p0 + 1e-9)

    def test_lambda_exactly_at_upper_bound_is_valid(self) -> None:
        """lambda = 1/p0 is the valid endpoint. LLD-C §3.1."""
        p0 = 0.5
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=0.05, lam=1.0 / p0)
        assert ep.wealth == pytest.approx(1.0)

    def test_lambda_zero_is_valid(self) -> None:
        """lambda = 0 (cash expert) is always valid."""
        ep = GraphEProcess.fixed_lambda(p0=0.8, alpha=0.05, lam=0.0)
        for _ in range(10):
            ep.update(1)
        assert ep.wealth == pytest.approx(1.0)  # cash: wealth never changes


# ── One-Step Factors — Exact Arithmetic ───────────────────────────────────


class TestOneStepFactors:
    """Exact one-step factor values — LLD-C Eq (3.2) and §14 tests #1–2."""

    def test_initial_wealth_is_one(self) -> None:
        """E_0 = 1 by definition — LLD-C Eq (3.2)."""
        ep = GraphEProcess.fixed_lambda(p0=0.8, alpha=0.05, lam=0.5)
        assert ep.wealth == pytest.approx(1.0)
        assert ep.log_wealth == pytest.approx(0.0)

    def test_success_factor_exact(self) -> None:
        """E_1(y=1) = 1 + lambda*(1-p0). With p0=0.8, lam=0.5: factor=1.1."""
        p0, lam = 0.8, 0.5
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=0.05, lam=lam)
        ep.update(1)
        expected = 1.0 + lam * (1.0 - p0)  # = 1.1
        assert ep.wealth == pytest.approx(expected, rel=1e-12)

    def test_failure_factor_exact(self) -> None:
        """E_1(y=0) = 1 - lambda*p0. With p0=0.8, lam=0.5: factor=0.6."""
        p0, lam = 0.8, 0.5
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=0.05, lam=lam)
        ep.update(0)
        expected = 1.0 - lam * p0  # = 0.6
        assert ep.wealth == pytest.approx(expected, rel=1e-12)

    def test_success_factor_at_lambda_endpoint_equals_1_over_p0(self) -> None:
        """1 + (1/p0)*(1-p0) = 1/p0. LLD-C §14 test #1."""
        p0 = 0.5
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=0.05, lam=1.0 / p0)
        ep.update(1)
        assert ep.wealth == pytest.approx(1.0 / p0, rel=1e-12)

    def test_failure_factor_zero_at_lambda_endpoint(self) -> None:
        """1 - (1/p0)*p0 = 0. LLD-C §14 test #2."""
        p0 = 0.5
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=0.05, lam=1.0 / p0)
        ep.update(0)
        assert ep.wealth == pytest.approx(0.0, abs=1e-15)

    def test_two_step_product_exact(self) -> None:
        """E_2 = E_1 * factor_2 multiplicatively. p0=0.8, lam=0.5: 1.1*0.6=0.66."""
        p0, lam = 0.8, 0.5
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=0.05, lam=lam)
        ep.update(1)  # factor = 1.1
        ep.update(0)  # factor = 0.6
        expected = (1.0 + lam * (1.0 - p0)) * (1.0 - lam * p0)  # = 0.66
        assert ep.wealth == pytest.approx(expected, rel=1e-12)


# ── Conditional Expected Factor (Supermartingale) ─────────────────────────


class TestConditionalExpectedFactor:
    """Supermartingale property — LLD-C Thm C.1 Eq (3.3), §14 tests #5–6."""

    def test_expected_factor_at_null_boundary_is_exactly_one(self) -> None:
        """E[1+lambda*(Y-p0)] = 1 when Y~Bernoulli(p0). LLD-C §14 test #5."""
        for p0, lam in [(0.6, 1.0), (0.7, 0.5), (0.3, 2.5), (0.5, 2.0)]:
            factor_s = 1.0 + lam * (1.0 - p0)
            factor_f = 1.0 - lam * p0
            expected_factor = p0 * factor_s + (1.0 - p0) * factor_f
            assert math.isclose(expected_factor, 1.0, rel_tol=1e-12), \
                f"p0={p0}, lam={lam}: expected 1.0, got {expected_factor}"

    def test_expected_factor_below_null_at_most_one(self) -> None:
        """E[factor] <= 1 for p_true <= p0. LLD-C §14 test #6."""
        p0, lam = 0.8, 0.5
        factor_s = 1.0 + lam * (1.0 - p0)
        factor_f = 1.0 - lam * p0
        for p_true in [0.0, 0.2, 0.4, 0.7, 0.79, 0.8]:
            e_factor = p_true * factor_s + (1.0 - p_true) * factor_f
            assert e_factor <= 1.0 + 1e-12, \
                f"p_true={p_true}: E[factor]={e_factor} > 1"

    def test_supermartingale_expected_wealth_at_p0_equals_prior_wealth(self) -> None:
        """With p_true = p0, E[E_r | F_{r-1}] = E_{r-1} exactly."""
        p0, lam = 0.7, 0.8
        # Enumerate all length-3 sequences under p0 exactly
        total_e3 = 0.0
        for seq in itertools.product([0, 1], repeat=3):
            prob = p0 ** sum(seq) * (1.0 - p0) ** (3 - sum(seq))
            wealth = 1.0
            for y in seq:
                wealth *= 1.0 + lam * (y - p0)
            total_e3 += prob * wealth
        # E[E_3] should equal E_0 = 1
        assert math.isclose(total_e3, 1.0, rel_tol=1e-10)


# ── Fixed-Lambda Sequential Process ───────────────────────────────────────


class TestFixedLambdaProcess:
    """Sequential updates and certification — LLD-C Thm C.1–C.2."""

    def test_mission_count_increments_each_update(self) -> None:
        ep = GraphEProcess.fixed_lambda(p0=0.8, alpha=0.05, lam=0.5)
        assert ep.mission_count == 0
        ep.update(1)
        assert ep.mission_count == 1
        ep.update(0)
        assert ep.mission_count == 2

    def test_update_returns_immutable_snapshot(self) -> None:
        ep = GraphEProcess.fixed_lambda(p0=0.8, alpha=0.05, lam=0.5)
        result = ep.update(1)
        assert isinstance(result, EProcessUpdate)
        assert result.mission_index == 1
        assert result.outcome == 1

    def test_certified_after_many_successes(self) -> None:
        """Wealth = 2^n; certified once 2^n >= 1/alpha. lam=1/p0 for fast crossing."""
        p0, alpha, lam = 0.5, 0.05, 2.0  # lam = 1/p0, factor=2 on success
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=alpha, lam=lam)
        # 2^4=16 < 20, 2^5=32 >= 20
        for _ in range(4):
            ep.update(1)
        assert not ep.certified()  # 16 < 20
        ep.update(1)
        assert ep.certified()  # 32 >= 20

    def test_cash_bet_never_certifies(self) -> None:
        """lambda=0 keeps E_r=1 forever; 1 < 1/alpha for alpha < 1."""
        ep = GraphEProcess.fixed_lambda(p0=0.8, alpha=0.05, lam=0.0)
        for _ in range(200):
            ep.update(1)
        assert not ep.certified()
        assert ep.wealth == pytest.approx(1.0)

    def test_first_crossing_index_none_before_threshold(self) -> None:
        p0, alpha, lam = 0.5, 0.05, 2.0
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=alpha, lam=lam)
        for _ in range(4):
            ep.update(1)
        assert ep.first_crossing_index is None

    def test_first_crossing_index_recorded_at_correct_mission(self) -> None:
        """Crossing index = 5 since 2^5=32 >= 20 = 1/alpha."""
        p0, alpha, lam = 0.5, 0.05, 2.0
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=alpha, lam=lam)
        for _ in range(4):
            ep.update(1)
        assert ep.first_crossing_index is None
        ep.update(1)  # 5th success
        assert ep.first_crossing_index == 5

    def test_optional_continuation_retains_crossing_index(self) -> None:
        """Continuing after first crossing does not reset crossing index — LLD-C §4.2."""
        p0, alpha, lam = 0.5, 0.05, 2.0
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=alpha, lam=lam)
        for _ in range(5):
            ep.update(1)
        crossing = ep.first_crossing_index
        assert crossing is not None
        # Continue with failures — wealth can fall but index must not change
        for _ in range(20):
            ep.update(0)
        assert ep.first_crossing_index == crossing

    def test_invalid_outcome_raises(self) -> None:
        ep = GraphEProcess.fixed_lambda(p0=0.8, alpha=0.05, lam=0.5)
        with pytest.raises(EProcessError):
            ep.update(2)


# ── SPRT Recovery — Exact LR Factors ──────────────────────────────────────


class TestSPRTRecovery:
    """lambda* recovers Bernoulli SPRT likelihood ratio — LLD-C §4.3, Eq (4.8)–(4.10)."""

    def test_success_factor_equals_p1_over_p0(self) -> None:
        """1 + lambda*(1-p0) = p1/p0. LLD-C Eq (4.9) y=1."""
        p0, p1 = 0.6, 0.9
        ep = GraphEProcess.from_sprt(p0=p0, alpha=0.05, p1=p1)
        ep.update(1)
        assert ep.wealth == pytest.approx(p1 / p0, rel=1e-12)

    def test_failure_factor_equals_one_minus_p1_over_one_minus_p0(self) -> None:
        """1 - lambda*p0 = (1-p1)/(1-p0). LLD-C Eq (4.9) y=0."""
        p0, p1 = 0.6, 0.9
        ep = GraphEProcess.from_sprt(p0=p0, alpha=0.05, p1=p1)
        ep.update(0)
        assert ep.wealth == pytest.approx((1.0 - p1) / (1.0 - p0), rel=1e-12)

    def test_product_matches_closed_form_lr_all_sequences_length_6(self) -> None:
        """E_r = (p1/p0)^k * ((1-p1)/(1-p0))^(r-k). LLD-C Eq (4.10), §14 test #15."""
        p0, p1 = 0.6, 0.85
        for r in range(1, 7):
            for seq in itertools.product([0, 1], repeat=r):
                ep = GraphEProcess.from_sprt(p0=p0, alpha=0.05, p1=p1)
                for y in seq:
                    ep.update(y)
                k = sum(seq)
                expected = (p1 / p0) ** k * ((1.0 - p1) / (1.0 - p0)) ** (r - k)
                assert ep.wealth == pytest.approx(expected, rel=1e-10), \
                    f"seq={seq}: expected {expected}, got {ep.wealth}"

    def test_lambda_star_formula(self) -> None:
        """lambda* = (p1-p0)/(p0*(1-p0)) — LLD-C Eq (4.8)."""
        p0, p1 = 0.4, 0.8
        expected_lam = (p1 - p0) / (p0 * (1.0 - p0))
        # Verify indirectly: success factor should equal p1/p0
        ep = GraphEProcess.from_sprt(p0=p0, alpha=0.05, p1=p1)
        ep.update(1)
        assert ep.wealth == pytest.approx(p1 / p0, rel=1e-12)
        # Compute lambda* manually and check against expected
        assert expected_lam == pytest.approx((p1 - p0) / (p0 * (1.0 - p0)), rel=1e-12)


# ── Mixture of Experts ────────────────────────────────────────────────────


class TestMixtureEProcess:
    """Mixture-of-experts e-process — LLD-C Thm C.3, Eq (5.14)–(5.21)."""

    def test_initial_mixture_wealth_is_one(self) -> None:
        ep = GraphEProcess.mixture(p0=0.7, alpha=0.05, epsilon=0.1)
        assert ep.wealth == pytest.approx(1.0)
        assert ep.log_wealth == pytest.approx(0.0)

    def test_cash_expert_keeps_wealth_positive(self) -> None:
        """E_r^mix >= pi_0 > 0 always. LLD-C Eq (5.14), Thm C.3."""
        ep = GraphEProcess.mixture(
            p0=0.5, alpha=0.05, epsilon=0.1,
            prior_weights=[0.4, 0.6],
        )
        pi_0 = ep.prior_weights[0]  # = 0.4
        for _ in range(20):
            ep.update(0)  # all failures → drives wealth down
        assert ep.wealth >= pi_0 - 1e-12

    def test_mixture_aggregate_bet_exact(self) -> None:
        """E_r^mix = E_{r-1}^mix * (1+lambda_bar*(y-p0)). LLD-C Eq (5.19)."""
        p0, eps = 0.3, 0.1
        ep = GraphEProcess.mixture(p0=p0, alpha=0.05, epsilon=eps)
        w_before = ep.wealth
        q_bar = ep.compute_aggregate_forecast()
        lam_bar = (q_bar - p0) / (p0 * (1.0 - p0))
        y = 1
        expected_wealth = w_before * (1.0 + lam_bar * (y - p0))
        ep.update(y)
        assert ep.wealth == pytest.approx(expected_wealth, rel=1e-12)

    def test_log_regret_bound_pathwise(self) -> None:
        """log E_n^mix >= log E_n^(k) - log(1/pi_k). LLD-C Eq (5.20), §14 test #25."""
        p0, eps = 0.4, 0.15
        for n in range(1, 9):
            for seq in itertools.product([0, 1], repeat=n):
                ep = GraphEProcess.mixture(
                    p0=p0, alpha=0.05, epsilon=eps,
                    extra_qs=[0.6],  # one extra fixed expert
                    prior_weights=[1.0 / 3.0] * 3,
                )
                for y in seq:
                    ep.update(y)
                log_mix = ep.log_wealth
                for k, (lw_k, pi_k) in enumerate(
                    zip(ep.expert_log_wealths, ep.prior_weights, strict=False)
                ):
                    assert log_mix >= lw_k + math.log(pi_k) - 1e-10, (
                        f"seq={seq}, k={k}: log_mix={log_mix:.4f} < "
                        f"lw_k + log(pi_k) = {lw_k + math.log(pi_k):.4f}"
                    )

    def test_logsumexp_cross_check(self) -> None:
        """Mixture log wealth = logsumexp_k(log(pi_k)+lw_k). LLD-C §14 test #35."""
        p0, eps = 0.3, 0.1
        ep = GraphEProcess.mixture(p0=p0, alpha=0.05, epsilon=eps)
        for y in [1, 1, 0, 1, 0, 0, 1]:
            ep.update(y)
        # log E_n^mix = log Σ_k pi_k * exp(lw_k)
        lse = math.log(
            sum(
                pi * math.exp(lw)
                for pi, lw in zip(ep.prior_weights, ep.expert_log_wealths, strict=False)
            )
        )
        assert ep.log_wealth == pytest.approx(lse, abs=1e-9)

    def test_terminal_only_forecast_no_off_by_one(self) -> None:
        """q_{1,r} = clip((1+sum_{s<r} Y)/(r+1), p0, 1-eps). LLD-C Eq (5.6), §14 test #39."""
        p0, eps = 0.2, 0.1
        ep = GraphEProcess.mixture(p0=p0, alpha=0.05, epsilon=eps)
        # Before any mission: r=0, success=0 → (1+0)/(0+2) = 0.5
        q0 = ep.terminal_only_forecast()
        assert q0 == pytest.approx(
            min(1.0 - eps, max(p0, 1.0 / 2.0)), rel=1e-12
        )
        # After 2 successes
        ep.update(1)
        ep.update(1)
        # r=2, success_sum=2 → (1+2)/(2+2) = 0.75
        q2 = ep.terminal_only_forecast()
        assert q2 == pytest.approx(
            min(1.0 - eps, max(p0, 3.0 / 4.0)), rel=1e-12
        )
        # After 1 failure
        ep.update(0)
        # r=3, success_sum=2 → (1+2)/(3+2) = 3/5=0.6
        q3 = ep.terminal_only_forecast()
        assert q3 == pytest.approx(
            min(1.0 - eps, max(p0, 3.0 / 5.0)), rel=1e-12
        )

    def test_expert_shares_sum_to_one(self) -> None:
        """Σ_k w_k = 1 after any sequence of updates."""
        ep = GraphEProcess.mixture(
            p0=0.5, alpha=0.05, epsilon=0.1,
            extra_qs=[0.65, 0.75],
            prior_weights=[0.25, 0.25, 0.25, 0.25],
        )
        for y in [1, 0, 1, 1, 0, 1, 0, 0]:
            ep.update(y)
        assert sum(ep.expert_shares) == pytest.approx(1.0, abs=1e-12)


# ── Leakage Counterexample ─────────────────────────────────────────────────


class TestLeakageCounterexample:
    """Current-mission leakage breaks the supermartingale — LLD-C Eq (5.31)."""

    def test_outcome_dependent_lambda_expected_factor_is_two_minus_p0(self) -> None:
        """With lambda_r = (1/p0)*1{Y=1} (leakage), E[factor] = 2-p0 > 1.

        This is the impossibility proof of LLD-C §5.6 Eq (5.31), showing why
        predictable bets must not depend on the current mission outcome.
        """
        p0 = 0.6
        # Factor when Y=1, leaky lambda = 1/p0: 1 + (1/p0)*(1-p0)
        factor_success = 1.0 + (1.0 / p0) * (1.0 - p0)
        # Factor when Y=0, leaky lambda = 0: 1 + 0*(0-p0) = 1
        factor_failure = 1.0
        expected = p0 * factor_success + (1.0 - p0) * factor_failure
        # Must equal 2-p0 = 1.4
        assert expected == pytest.approx(2.0 - p0, rel=1e-12)
        assert expected > 1.0  # violates supermartingale condition


# ── KL Divergence Utility ─────────────────────────────────────────────────


class TestKLBernoulli:
    """Bernoulli KL divergence — LLD-C Eq (5.22)."""

    def test_kl_of_p_with_itself_is_zero(self) -> None:
        for p in [0.1, 0.5, 0.9]:
            assert kl_bernoulli(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_kl_known_value(self) -> None:
        """d(0.8 || 0.6) computed by hand."""
        p, q = 0.8, 0.6
        expected = p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
        assert kl_bernoulli(p, q) == pytest.approx(expected, rel=1e-12)

    def test_kl_is_nonnegative(self) -> None:
        rng = random.Random(42)
        for _ in range(50):
            p = rng.uniform(0.01, 0.99)
            q = rng.uniform(0.01, 0.99)
            assert kl_bernoulli(p, q) >= -1e-12

    def test_kl_growth_identity(self) -> None:
        """E[log_factor | Y~Bernoulli(p)] = d(p||p0) - d(p||q). LLD-C Eq (5.21)."""
        rng = random.Random(99)
        p0 = 0.4
        for _ in range(30):
            p = rng.uniform(0.01, 0.99)
            q = rng.uniform(0.01, 0.99)
            # Expected log growth for expert forecasting q under null p0
            el_growth = p * math.log(q / p0) + (1.0 - p) * math.log(
                (1.0 - q) / (1.0 - p0)
            )
            identity = kl_bernoulli(p, p0) - kl_bernoulli(p, q)
            assert el_growth == pytest.approx(identity, rel=1e-9, abs=1e-12)


# ── Type-I Error Control (Ville's Theorem) ────────────────────────────────


class TestVilleTypeI:
    """Empirical crossing rate <= alpha — LLD-C Thm C.2 Eq (4.2)."""

    def test_null_crossing_rate_at_most_alpha_fixed_lambda(self) -> None:
        """Over 10k seeded null streams, empirical type-I rate <= alpha + MC noise."""
        p0, alpha = 0.7, 0.05
        lam = 0.7 / p0  # in (0, 1/p0), closer to endpoint for faster growth
        rate = simulate_type1_crossing_rate(
            p0=p0,
            alpha=alpha,
            fixed_lambda=lam,
            n_streams=10_000,
            n_missions=300,
            rng_seed=42,
        )
        # Ville guarantees rate <= alpha; allow 3-sigma Monte Carlo tolerance:
        # 3*sqrt(0.05*0.95/10000) ≈ 0.0065
        assert rate <= alpha + 0.01, (
            f"Type-I rate {rate:.4f} exceeded alpha+tolerance {alpha+0.01:.4f}"
        )

    def test_null_crossing_rate_at_boundary_p0(self) -> None:
        """p_true = p0 (boundary of null); still must satisfy Ville's bound."""
        p0, alpha = 0.5, 0.05
        lam = 1.5  # valid: 1.5 < 1/0.5 = 2.0
        rate = simulate_type1_crossing_rate(
            p0=p0,
            alpha=alpha,
            fixed_lambda=lam,
            n_streams=8_000,
            n_missions=300,
            rng_seed=123,
        )
        assert rate <= alpha + 0.015


# ── Certificate Latching (LLD-C §4.3 tau_alpha one-way declaration) ────────


class TestCertificateLatch:
    """certified() is a one-way declaration — once True it must stay True.

    LLD-C §4.3: the certificate is issued at the first crossing time tau_alpha.
    Ville controls type-I error over the stopping time, so the certificate must
    latch: max log-wealth ever seen, not current wealth, determines certification.
    """

    def test_certified_latches_after_crossing_then_falling(self) -> None:
        """certified() stays True once E_r >= 1/alpha, even if wealth later falls.

        Uses lam=1.5 < 1/p0=2.0 so failures do not absorb (factor=0.25 > 0).
        After 7 successes wealth ≈ 20.5 >= 20 = 1/alpha; 20 failures drive it
        near zero, but certified() must remain True.
        """
        p0, alpha, lam = 0.5, 0.05, 1.5
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=alpha, lam=lam)
        # success factor = 1 + 1.5*0.5 = 1.75; 1.75^7 ≈ 20.5 >= 20 = 1/alpha
        for _ in range(7):
            ep.update(1)
        assert ep.certified(), (
            f"Expected certified after 7 successes, log_wealth={ep.log_wealth:.3f}"
        )
        # 20 failures: failure factor = 1 - 1.5*0.5 = 0.25 per step
        for _ in range(20):
            ep.update(0)
        # log_wealth should have fallen well below threshold
        threshold = -math.log(alpha)
        assert ep.log_wealth < threshold, (
            "Wealth did not fall below threshold — test setup wrong"
        )
        # certified() must still be True (latched at first crossing)
        assert ep.certified(), (
            f"certified() must stay True after first crossing: "
            f"log_wealth={ep.log_wealth:.3f} < threshold={threshold:.3f}"
        )

    def test_certified_responds_to_alpha_argument(self) -> None:
        """certified(alpha) uses the given alpha to set the threshold.

        After 4 successes, wealth = 1.75^4 ≈ 9.4:
          certified(alpha=0.10) → threshold=10 → False (9.4 < 10)
          certified(alpha=0.20) → threshold=5  → True (9.4 >= 5)

        After 5 successes, wealth = 1.75^5 ≈ 16.4:
          certified(alpha=0.10) → True (16.4 >= 10)
          certified(alpha=0.05) → False (16.4 < 20)

        After 7 successes, wealth = 1.75^7 ≈ 20.5 → certified at all tested levels.
        """
        p0, alpha_stored, lam = 0.5, 0.05, 1.5
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=alpha_stored, lam=lam)

        # 4 successes: 1.75^4 ≈ 9.4
        for _ in range(4):
            ep.update(1)
        assert ep.certified(alpha=0.20), "1.75^4≈9.4 >= 1/0.20=5 → should be True"
        assert not ep.certified(alpha=0.10), "1.75^4≈9.4 < 1/0.10=10 → should be False"

        # 7 successes total: 1.75^7 ≈ 20.5
        for _ in range(3):
            ep.update(1)
        assert ep.certified(alpha=0.10), "1.75^7≈20.5 >= 1/0.10=10 → True"
        assert ep.certified(alpha=0.05), "1.75^7≈20.5 >= 1/0.05=20 → True"
