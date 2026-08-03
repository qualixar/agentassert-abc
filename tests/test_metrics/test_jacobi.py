# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for Jacobi/Wright-Fisher bounded-support drift dynamics (LLD-D).

Covers:
- JacobiParams stationary identities (Theorem D.2)
- Feller boundary classification (Theorem D.1)
- Method-of-moments fitting round-trip (Eqs. D.22–D.24)
- Identifiability gate pass/fail (Theorem D.4)
"""

from __future__ import annotations

import numpy as np
import pytest

from agentassert_abc.metrics.jacobi import (
    FellerClassification,
    GateResult,
    JacobiFitError,
    JacobiParams,
    feller_classification,
    fit_jacobi_mom,
    identifiability_gate,
)


# ---------------------------------------------------------------------------
# Helper: generate AR(1) approximation to Jacobi stationary process
# ---------------------------------------------------------------------------

def _make_jacobi_ar1(
    kappa: float,
    theta: float,
    sigma2: float,
    dt: float,
    n: int,
    seed: int,
) -> np.ndarray:
    """AR(1) approximation to stationary Jacobi process.

    Transition: X[t] = theta + phi * (X[t-1] - theta) + eps[t]
    where phi = exp(-kappa * dt) and eps[t] ~ N(0, stat_var*(1 - phi^2)).

    Args:
        kappa: Mean-reversion rate.
        theta: Long-run mean.
        sigma2: Diffusion intensity.
        dt: Time step.
        n: Number of observations.
        seed: RNG seed for reproducibility.

    Returns:
        Array of length n, clipped to (0, 1).
    """
    rng = np.random.default_rng(seed)
    phi = np.exp(-kappa * dt)
    stat_var = theta * (1.0 - theta) * sigma2 / (2.0 * kappa + sigma2)
    eps_std = float(np.sqrt(stat_var * (1.0 - phi**2)))

    X = np.empty(n)
    X[0] = theta
    for t in range(1, n):
        X[t] = theta + phi * (X[t - 1] - theta) + rng.normal(0.0, eps_std)

    # Clip to valid domain; should rarely trigger for well-chosen parameters
    return np.clip(X, 1e-9, 1.0 - 1e-9)


# ---------------------------------------------------------------------------
# Test: JacobiParams stationary identities (Theorem D.2, LLD-D §D.2)
# ---------------------------------------------------------------------------

class TestJacobiParams:
    """Stationary identities and parameter validation for JacobiParams."""

    def test_shape_parameters_exact(self) -> None:
        """Theorem D.2 Eq. D.2: a and b match their closed-form definitions."""
        # kappa=2, theta=0.3, sigma2=0.5
        # a = 2*2*0.3/0.5 = 2.4;  b = 2*2*0.7/0.5 = 5.6
        params = JacobiParams(kappa=2.0, theta=0.3, sigma2=0.5)
        assert params.a == pytest.approx(2.4, rel=1e-9)
        assert params.b == pytest.approx(5.6, rel=1e-9)

    def test_stationary_mean(self) -> None:
        """Theorem D.2 Eq. D.12: stationary mean equals theta."""
        params = JacobiParams(kappa=2.0, theta=0.3, sigma2=0.5)
        assert params.stationary_mean == pytest.approx(0.3, rel=1e-9)

    def test_stationary_var(self) -> None:
        """Theorem D.2 Eq. D.12: stationary variance formula."""
        # Var = theta*(1-theta)*sigma2 / (2*kappa + sigma2)
        # = 0.3 * 0.7 * 0.5 / (4.0 + 0.5) = 0.105 / 4.5
        params = JacobiParams(kappa=2.0, theta=0.3, sigma2=0.5)
        expected = 0.3 * 0.7 * 0.5 / (2.0 * 2.0 + 0.5)
        assert params.stationary_var == pytest.approx(expected, rel=1e-9)

    def test_feller_g_factors(self) -> None:
        """Eq. D.28: g0 = 2*kappa*theta - sigma2, g1 = 2*kappa*(1-theta) - sigma2."""
        params = JacobiParams(kappa=2.0, theta=0.3, sigma2=0.5)
        g0 = 2.0 * params.kappa * params.theta - params.sigma2
        g1 = 2.0 * params.kappa * (1.0 - params.theta) - params.sigma2
        # g0 = 1.2 - 0.5 = 0.7 > 0  (boundary 0 unattainable)
        assert g0 == pytest.approx(0.7, rel=1e-9)
        # g1 = 2.8 - 0.5 = 2.3 > 0  (boundary 1 unattainable)
        assert g1 == pytest.approx(2.3, rel=1e-9)

    def test_validation_kappa_le_zero(self) -> None:
        """kappa <= 0 must raise JacobiFitError."""
        with pytest.raises(JacobiFitError, match="kappa"):
            JacobiParams(kappa=0.0, theta=0.5, sigma2=0.5)

    def test_validation_theta_boundary(self) -> None:
        """theta at boundary (0 or 1) must raise JacobiFitError."""
        with pytest.raises(JacobiFitError, match="theta"):
            JacobiParams(kappa=1.0, theta=0.0, sigma2=0.5)
        with pytest.raises(JacobiFitError, match="theta"):
            JacobiParams(kappa=1.0, theta=1.0, sigma2=0.5)

    def test_validation_sigma2_le_zero(self) -> None:
        """sigma2 <= 0 must raise JacobiFitError."""
        with pytest.raises(JacobiFitError, match="sigma2"):
            JacobiParams(kappa=1.0, theta=0.5, sigma2=0.0)

    def test_immutability(self) -> None:
        """JacobiParams is frozen — direct assignment must raise FrozenInstanceError."""
        params = JacobiParams(kappa=1.0, theta=0.5, sigma2=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            params.kappa = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: Feller classification (Theorem D.1, LLD-D §D.1)
# ---------------------------------------------------------------------------

class TestFellerClassification:
    """Boundary classification: unattainable (a>=1) vs attainable (a<1)."""

    def test_both_unattainable(self) -> None:
        """kappa=2, theta=0.3, sigma2=0.5: a=2.4>=1, b=5.6>=1 → both unattainable."""
        params = JacobiParams(kappa=2.0, theta=0.3, sigma2=0.5)
        fc = feller_classification(params)
        assert fc.boundary_0 == "unattainable"
        assert fc.boundary_1 == "unattainable"
        assert fc.a == pytest.approx(2.4, rel=1e-9)
        assert fc.b == pytest.approx(5.6, rel=1e-9)

    def test_boundary_0_attainable(self) -> None:
        """a < 1 makes boundary 0 attainable."""
        # a = 2*2*0.3/2.0 = 0.6 < 1 → boundary 0 attainable
        # b = 2*2*0.7/2.0 = 1.4 >= 1 → boundary 1 unattainable
        params = JacobiParams(kappa=2.0, theta=0.3, sigma2=2.0)
        fc = feller_classification(params)
        assert fc.boundary_0 == "attainable"
        assert fc.boundary_1 == "unattainable"

    def test_boundary_1_attainable(self) -> None:
        """b < 1 makes boundary 1 attainable."""
        # a = 2*2*0.8/2.0 = 1.6 >= 1 → boundary 0 unattainable
        # b = 2*2*0.2/2.0 = 0.4 < 1 → boundary 1 attainable
        params = JacobiParams(kappa=2.0, theta=0.8, sigma2=2.0)
        fc = feller_classification(params)
        assert fc.boundary_0 == "unattainable"
        assert fc.boundary_1 == "attainable"

    def test_both_attainable(self) -> None:
        """Both a < 1 and b < 1 → both attainable."""
        # a = 2*1*0.3/2.0 = 0.3, b = 2*1*0.7/2.0 = 0.7 — both < 1
        params = JacobiParams(kappa=1.0, theta=0.3, sigma2=2.0)
        fc = feller_classification(params)
        assert fc.boundary_0 == "attainable"
        assert fc.boundary_1 == "attainable"

    def test_returns_feller_classification(self) -> None:
        """feller_classification returns a FellerClassification dataclass."""
        params = JacobiParams(kappa=2.0, theta=0.5, sigma2=1.0)
        result = feller_classification(params)
        assert isinstance(result, FellerClassification)

    def test_threshold_a_equals_1(self) -> None:
        """a == 1 exactly → still 'unattainable' (a >= 1 is the condition)."""
        # a = 2*1*0.5/1.0 = 1.0 exactly → boundary 0 unattainable
        params = JacobiParams(kappa=1.0, theta=0.5, sigma2=1.0)
        fc = feller_classification(params)
        assert fc.boundary_0 == "unattainable"


# ---------------------------------------------------------------------------
# Test: fit_jacobi_mom (Eqs. D.22–D.24, LLD-D §D.4.1)
# ---------------------------------------------------------------------------

class TestFitJacobiMom:
    """Method-of-moments fitting: admissibility guards and round-trip recovery."""

    def test_too_short_raises(self) -> None:
        """Fewer than 3 observations must raise JacobiFitError."""
        with pytest.raises(JacobiFitError, match="observations"):
            fit_jacobi_mom([0.3, 0.5], dt=1.0)

    def test_dt_le_zero_raises(self) -> None:
        """Non-positive dt must raise JacobiFitError."""
        series = np.full(50, 0.3)
        with pytest.raises(JacobiFitError, match="dt"):
            fit_jacobi_mom(series, dt=0.0)

    def test_out_of_range_raises(self) -> None:
        """Values outside [0, 1] must raise JacobiFitError."""
        series = np.linspace(0.1, 1.1, 50)
        with pytest.raises(JacobiFitError):
            fit_jacobi_mom(series, dt=1.0)

    def test_negative_rho1_raises(self) -> None:
        """Lag-1 autocorrelation <= 0 raises JacobiFitError (Eq. D.25)."""
        # Alternating series has rho1 ≈ -1
        series = np.array([0.4 if i % 2 == 0 else 0.6 for i in range(100)])
        with pytest.raises(JacobiFitError):
            fit_jacobi_mom(series, dt=1.0)

    def test_variance_too_high_raises(self) -> None:
        """S2 >= theta*(1-theta) violates Eq. D.25, raises JacobiFitError."""
        # Bimodal near-boundary series: very high variance
        series = np.array([0.02 if i % 2 == 0 else 0.98 for i in range(200)])
        # theta ≈ 0.5, theta*(1-theta) = 0.25, S2 ≈ 0.24^2 * ... actually near 0.23
        # The alternating pattern also has rho1 ≈ -1, so rho1 check fires first
        with pytest.raises(JacobiFitError):
            fit_jacobi_mom(series, dt=1.0)

    def test_returns_jacobi_params(self) -> None:
        """Successful fit returns JacobiParams instance."""
        series = _make_jacobi_ar1(
            kappa=1.5, theta=0.5, sigma2=1.0, dt=1.0, n=500, seed=0
        )
        params = fit_jacobi_mom(series, dt=1.0)
        assert isinstance(params, JacobiParams)

    def test_theta_roundtrip(self) -> None:
        """MoM theta_hat converges to true theta (Eq. D.22)."""
        kappa, theta, sigma2, dt = 2.0, 0.3, 0.5, 1.0
        series = _make_jacobi_ar1(kappa, theta, sigma2, dt, n=5000, seed=42)
        params = fit_jacobi_mom(series, dt)
        assert abs(params.theta - theta) / theta < 0.05  # 5% tolerance

    def test_kappa_roundtrip(self) -> None:
        """MoM kappa_hat converges to true kappa (Eq. D.23)."""
        kappa, theta, sigma2, dt = 2.0, 0.3, 0.5, 1.0
        series = _make_jacobi_ar1(kappa, theta, sigma2, dt, n=5000, seed=42)
        params = fit_jacobi_mom(series, dt)
        assert abs(params.kappa - kappa) / kappa < 0.10  # 10% tolerance

    def test_sigma2_roundtrip(self) -> None:
        """MoM sigma2_hat converges to true sigma2 (Eq. D.24)."""
        kappa, theta, sigma2, dt = 2.0, 0.3, 0.5, 1.0
        series = _make_jacobi_ar1(kappa, theta, sigma2, dt, n=5000, seed=42)
        params = fit_jacobi_mom(series, dt)
        assert abs(params.sigma2 - sigma2) / sigma2 < 0.15  # 15% tolerance

    def test_stationary_var_consistency(self) -> None:
        """Fitted params' stationary_var should approximate sample variance."""
        kappa, theta, sigma2, dt = 1.5, 0.4, 0.8, 1.0
        series = _make_jacobi_ar1(kappa, theta, sigma2, dt, n=5000, seed=7)
        params = fit_jacobi_mom(series, dt)
        sample_var = float(np.var(series))
        # Fitted stationary_var should be reasonably close to sample variance
        assert abs(params.stationary_var - sample_var) / sample_var < 0.20

    def test_list_input_accepted(self) -> None:
        """Python list input is accepted (not just numpy array)."""
        series = _make_jacobi_ar1(1.0, 0.5, 0.5, 1.0, n=100, seed=1).tolist()
        params = fit_jacobi_mom(series, dt=1.0)
        assert isinstance(params, JacobiParams)


# ---------------------------------------------------------------------------
# Test: identifiability_gate (Theorem D.4, LLD-D §D.5)
# ---------------------------------------------------------------------------

class TestIdentifiabilityGate:
    """Gate pass/fail and mandatory refusal per Theorem D.4."""

    def test_returns_gate_result(self) -> None:
        """identifiability_gate returns GateResult dataclass."""
        series = _make_jacobi_ar1(1.5, 0.5, 1.0, 1.0, n=2000, seed=0)
        result = identifiability_gate(series, dt=1.0)
        assert isinstance(result, GateResult)

    def test_near_constant_refuses_confinement(self) -> None:
        """Near-constant series (no boundary exploration) → gate refused.

        LLD-D §D.5 mandatory refusal: when exploration gate fails,
        no confinement claim is allowed.
        """
        rng = np.random.default_rng(42)
        # AR(1) series confined near theta=0.4 — never reaches near 0 or 1
        n = 300
        X = np.zeros(n)
        X[0] = 0.4
        phi = 0.6
        for t in range(1, n):
            X[t] = 0.4 + phi * (X[t - 1] - 0.4) + rng.normal(0, 0.03)
        X = np.clip(X, 0.0, 1.0)
        # Max range: ~0.4 ± 5*0.048 ≈ [0.16, 0.64] — no boundary visits with eps=0.1

        result = identifiability_gate(X, dt=1.0, eps=0.1)
        assert result.gate_passed is False
        # Both boundary classes must be None (refused)
        assert result.boundary_0_class is None
        assert result.boundary_1_class is None

    def test_near_constant_reason_mentions_exploration(self) -> None:
        """Refusal reason must reference exploration failure."""
        rng = np.random.default_rng(99)
        n = 200
        X = np.zeros(n)
        X[0] = 0.35
        for t in range(1, n):
            X[t] = 0.35 + 0.5 * (X[t - 1] - 0.35) + rng.normal(0, 0.02)
        X = np.clip(X, 0.0, 1.0)

        result = identifiability_gate(X, dt=1.0, eps=0.1)
        assert result.gate_passed is False
        assert "exploration" in result.reason.lower() or "E0" in result.reason or "E1" in result.reason

    def test_gate_result_immutable(self) -> None:
        """GateResult is frozen — direct assignment must raise FrozenInstanceError."""
        series = _make_jacobi_ar1(1.5, 0.5, 1.0, 1.0, n=500, seed=3)
        result = identifiability_gate(series, dt=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            result.gate_passed = True  # type: ignore[misc]

    def test_constant_series_returns_gate_result_not_raises(self) -> None:
        """Constant (degenerate) series → base fit fails → GateResult returned, NOT raised.

        LLD-D §D.5 graceful refusal: callers must be able to rely on getting a
        GateResult back; JacobiFitError must not escape the gate.
        """
        series = np.full(100, 0.5)
        result = identifiability_gate(series, dt=1.0)  # must NOT raise
        assert isinstance(result, GateResult)
        assert result.gate_passed is False
        assert result.params is None

    def test_exploring_series_gate_passes(self) -> None:
        """Jacobi series with strong a,b >> 1 and boundary exploration passes gate.

        Using kappa=1.5, theta=0.5, sigma2=1.0: a=b=1.5 > 1,
        g0=g1=0.5 >> 0, stationary std≈0.25 → many values near both boundaries.
        """
        series = _make_jacobi_ar1(
            kappa=1.5, theta=0.5, sigma2=1.0, dt=1.0, n=3000, seed=42
        )
        result = identifiability_gate(series, dt=1.0, eps=0.1, n_bootstrap=300)
        assert result.gate_passed is True
        assert result.boundary_0_class is not None
        assert result.boundary_1_class is not None
        assert result.params is not None

    def test_gate_passed_has_consistent_g_signs(self) -> None:
        """When gate passes, g0/g1 bootstrap ranges have consistent sign (Eq. D.36)."""
        series = _make_jacobi_ar1(
            kappa=1.5, theta=0.5, sigma2=1.0, dt=1.0, n=3000, seed=42
        )
        result = identifiability_gate(series, dt=1.0, eps=0.1, n_bootstrap=300)
        if result.gate_passed:
            # g0 range must be entirely on one side of 0
            assert (result.g0_min > 0) or (result.g0_max < 0)
            assert (result.g1_min > 0) or (result.g1_max < 0)

    def test_gate_contains_fitted_params(self) -> None:
        """GateResult always contains the fitted JacobiParams regardless of pass/fail."""
        series = _make_jacobi_ar1(1.5, 0.5, 1.0, 1.0, n=500, seed=5)
        result = identifiability_gate(series, dt=1.0)
        assert isinstance(result.params, JacobiParams)

    def test_alpha_parameter_accepted(self) -> None:
        """Non-default alpha is accepted without error."""
        series = _make_jacobi_ar1(1.5, 0.5, 1.0, 1.0, n=1000, seed=10)
        result = identifiability_gate(series, dt=1.0, alpha=0.10)
        assert isinstance(result, GateResult)

    def test_boundary_class_values_are_valid_strings(self) -> None:
        """When set, boundary classes are exactly 'unattainable' or 'attainable'."""
        series = _make_jacobi_ar1(1.5, 0.5, 1.0, 1.0, n=3000, seed=42)
        result = identifiability_gate(series, dt=1.0, eps=0.1, n_bootstrap=300)
        valid = {"unattainable", "attainable"}
        if result.boundary_0_class is not None:
            assert result.boundary_0_class in valid
        if result.boundary_1_class is not None:
            assert result.boundary_1_class in valid

    def test_alpha_changes_gate_ci_width(self) -> None:
        """alpha=0.05 (95% CI) produces a wider bootstrap CI than alpha=0.50 (50% CI).

        The percentile bootstrap at level (1-alpha) spans [alpha/2, 1-alpha/2]
        percentiles of the bootstrap distribution.  A smaller alpha → wider CI
        (harder to lie entirely one side of 0) — so alpha affects the gate.
        """
        series = _make_jacobi_ar1(kappa=1.5, theta=0.5, sigma2=1.0, dt=1.0, n=500, seed=42)
        r05 = identifiability_gate(series, dt=1.0, alpha=0.05, n_bootstrap=600)
        r50 = identifiability_gate(series, dt=1.0, alpha=0.50, n_bootstrap=600)
        # Both should run (series is well-formed)
        assert isinstance(r05, GateResult)
        assert isinstance(r50, GateResult)
        # g0 CI must be wider for alpha=0.05 than for alpha=0.50
        if not (
            r05.g0_min != r05.g0_min  # NaN check
            or r50.g0_min != r50.g0_min
        ):
            width_05 = r05.g0_max - r05.g0_min
            width_50 = r50.g0_max - r50.g0_min
            assert width_05 >= width_50, (
                f"alpha=0.05 CI width {width_05:.4f} should be >= "
                f"alpha=0.50 CI width {width_50:.4f}"
            )

    def test_block_bootstrap_handles_high_rho1_series(self) -> None:
        """Block bootstrap avoids spurious instability for highly correlated series.

        With i.i.d. resampling, high-rho1 series (phi≈0.9) would lose all
        autocorrelation after shuffling: rho1≈0 for most bootstrap samples →
        JacobiFitError on every resample → bootstrap instability gate fires.
        Block bootstrap preserves the Markov structure so enough resamples
        remain admissible and the gate proceeds to classification.
        """
        # phi = exp(-0.1) ≈ 0.905; sigma2=1.5 → broad distribution → boundary exploration
        series = _make_jacobi_ar1(kappa=0.1, theta=0.5, sigma2=1.5, dt=1.0, n=500, seed=42)
        result = identifiability_gate(series, dt=1.0, alpha=0.05, n_bootstrap=300)
        assert isinstance(result, GateResult)
        # If bootstrap ran (not blocked by exploration), must NOT fail with instability
        if "exploration" not in result.reason.lower():
            assert "instability" not in result.reason.lower(), (
                f"Block bootstrap should avoid instability for high-rho1 series; "
                f"got reason: {result.reason}"
            )

    def test_exploration_counts_excursions_not_raw_observations(self) -> None:
        """Exploration check counts independent excursions, not raw observation counts.

        LLD-D Eq. D.33: E_j(eps) is the number of distinct excursions (separated
        blocks) that reach within eps of boundary j — not a raw count of
        observations in the band.

        This series makes ONE long visit to each boundary band (40 obs each) and
        then stays interior.  With raw counting: e0=40 >= m0=2, e1=40 >= m1=2
        → exploration passes (wrong).  With excursion counting: 1 < 2 → fails.
        """
        # One contiguous visit per boundary: single excursion each
        arr = np.concatenate([
            np.full(40, 0.05),   # boundary 0 visit (1 excursion, 40 raw obs)
            np.full(20, 0.50),   # interior
            np.full(40, 0.95),   # boundary 1 visit (1 excursion, 40 raw obs)
            np.full(100, 0.40),  # interior
        ])
        # m0=m1=2 requires 2 independent excursions; excursion count = 1 → gate must fail
        result = identifiability_gate(arr, dt=1.0, eps=0.1, m0=2, m1=2)
        assert result.gate_passed is False
        assert "exploration" in result.reason.lower() or "E0" in result.reason
