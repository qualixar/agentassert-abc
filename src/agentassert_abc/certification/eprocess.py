# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Graph-level anytime-valid e-process certification — LLD-C Thm C.1–C.3.

Implements the sequential betting e-process (Ville 1939, Shafer & Vovk 2001,
Ramdas et al. 2023) for AgentAssert v2 graph-outcome certification.

Core guarantee: for every law P in the composite null P_0(p0) = {P : p_{G,r}
<= p0 a.s. for all r}, the process E_r is a nonnegative test supermartingale
(Thm C.1) and P(sup_r E_r >= 1/alpha) <= alpha uniformly (Thm C.2, Ville).

No independence assumption across missions, components, or shared-LLM shocks
is required; only the scalar conditional-mean null suffices (Thm C.5).

Modules:
    EProcessError   — validation and runtime errors.
    EProcessUpdate  — immutable snapshot returned by GraphEProcess.update().
    GraphEProcess   — main certifier; supports fixed-lambda and mixture modes.
    kl_bernoulli    — Bernoulli KL divergence utility (LLD-C Eq 5.22).
    simulate_type1_crossing_rate — empirical type-I calibration helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentassert_abc.exceptions import AgentAssertError

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class EProcessError(AgentAssertError):
    """Invalid configuration or illegal operation for a graph e-process.

    Raised for out-of-range parameters (LLD-C §8.1, §10), invalid lambda
    (LLD-C Eq 3.1), or illegal outcome values.
    """


# ---------------------------------------------------------------------------
# Immutable update snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EProcessUpdate:
    """Immutable state snapshot returned after one sequential update.

    Attributes:
        mission_index: Evidence index r (1-based).
        outcome: Y_{G,r} in {0, 1}.
        log_wealth: log E_r for the aggregate process.
        first_crossing_index: First r where E_r >= 1/alpha, or None.

    References:
        LLD-C §8.2 CertificateState.
    """

    mission_index: int
    outcome: int
    log_wealth: float
    first_crossing_index: int | None


# ---------------------------------------------------------------------------
# Mathematical utilities
# ---------------------------------------------------------------------------


def kl_bernoulli(p: float, q: float) -> float:
    """Bernoulli KL divergence d(p||q).

    d(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q)), with 0*log(0) = 0.

    Args:
        p: True probability in [0, 1].
        q: Reference probability in (0, 1).

    Returns:
        Non-negative KL divergence value.

    References:
        LLD-C Eq (5.22).
    """
    result = 0.0
    if p > 0.0:
        result += p * math.log(p / q)
    if p < 1.0:
        result += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return result


def _clip(q: float, p0: float, epsilon: float) -> float:
    """Clip forecast to [p0, 1-epsilon] — LLD-C Eq (5.1)."""
    return min(1.0 - epsilon, max(p0, q))


def _log_factor_lr(q: float, p0: float, y: int) -> float:
    """Log of the LR-shaped one-step factor — LLD-C §8.3.

    Uses log(q/p0) for y=1, log((1-q)/(1-p0)) for y=0.
    More numerically stable than log(1 + lambda*(y-p0)).

    Args:
        q: Expert forecast in [p0, 1].
        p0: Null threshold.
        y: Outcome in {0, 1}.

    Returns:
        Log factor; -inf if factor is zero (wealth absorbed).
    """
    if y == 1:
        return math.log(q) - math.log(p0)
    return math.log1p(-q) - math.log1p(-p0)


def _log_factor_fixed(lam: float, p0: float, y: int) -> float:
    """Log of the one-step factor for a constant lambda — LLD-C Eq (3.2).

    Args:
        lam: Betting fraction in [0, 1/p0].
        p0: Null threshold.
        y: Outcome in {0, 1}.

    Returns:
        Log factor; -inf at the boundary y=0, lam=1/p0.
    """
    factor = 1.0 + lam * (1.0 - p0) if y == 1 else 1.0 - lam * p0
    if factor <= 0.0:
        return -math.inf
    return math.log(factor)


def _logsumexp(log_vals: list[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not log_vals:
        return -math.inf
    max_val = max(log_vals)
    if math.isinf(max_val) and max_val < 0.0:
        return -math.inf
    return max_val + math.log(sum(math.exp(v - max_val) for v in log_vals))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_p0_alpha(p0: float, alpha: float) -> None:
    """Raise EProcessError if p0 or alpha are outside (0, 1) — LLD-C C-A1."""
    if not (0.0 < p0 < 1.0):
        raise EProcessError(
            f"p0 must be in (0, 1), got {p0}. "
            "LLD-C §8.1 requires an interior target (C-A1)."
        )
    if not (0.0 < alpha < 1.0):
        raise EProcessError(
            f"alpha must be in (0, 1), got {alpha}. "
            "LLD-C §8.1 rejects boundary significance levels."
        )


def _validate_lambda(lam: float, p0: float) -> None:
    """Raise EProcessError if lambda is outside [0, 1/p0] — LLD-C Eq (3.1).

    The interval [0, 1/p0] is the maximal interval for which:
    - 1 - lambda*p0 >= 0 (failure factor nonnegative), and
    - negative lambda does not violate the one-sided-null supermartingale.

    References:
        LLD-C Thm C.1, §3.1, §11.1.
    """
    if lam < 0.0:
        raise EProcessError(
            f"lambda must be >= 0 (negative bets violate the one-sided null "
            f"supermartingale — LLD-C §3.1), got {lam}"
        )
    upper = 1.0 / p0
    # Small absolute tolerance to allow exact boundary lam = 1/p0 in float arithmetic
    # while still rejecting lam = 1/p0 + 1e-9 (absolute, not relative, offset)
    if lam > upper + 1e-10:
        raise EProcessError(
            f"lambda must be <= 1/p0 = {upper:.6g} "
            f"(above this the failure factor is negative — LLD-C Thm C.1), got {lam}"
        )


# ---------------------------------------------------------------------------
# GraphEProcess
# ---------------------------------------------------------------------------


class GraphEProcess:
    """Graph-level anytime-valid e-process certifier — LLD-C Thm C.1–C.3.

    Two operating modes:

    **Fixed-lambda** (``fixed_lambda``, ``from_sprt``):
        Single constant bet lambda per mission.  The Wald-SPRT special case
        ``from_sprt(p0, p1)`` sets lambda* = (p1-p0)/(p0*(1-p0)) and
        recovers the Bernoulli likelihood-ratio factor exactly (LLD-C §4.3
        Eq 4.8–4.10).

    **Mixture** (``mixture``):
        Wealth-weighted aggregation of K >= 2 predictable experts: cash
        (q=p0), terminal-only Laplace-smoothed baseline, and optional
        fixed-forecast experts.  Achieves the pathwise log-regret bound
        log E_n^mix >= log E_n^(k) - log(1/pi_k) (Thm C.3 Eq 5.20).

    All bets are predictable (F_{r-1}-measurable) and the mixture log wealth
    is updated via the exact aggregate-bet formula (Thm C.3 Eq 5.19).

    Usage::

        ep = GraphEProcess.fixed_lambda(p0=0.8, alpha=0.05, lam=0.5)
        for y in mission_outcomes:
            update = ep.update(y)
        if ep.certified():
            print(f"Certified at mission {ep.first_crossing_index}")

    References:
        LLD-C §3 (Thm C.1), §4 (Thm C.2), §5 (Thm C.3), §8 (algorithm).
    """

    _MODE_FIXED = "fixed"
    _MODE_MIXTURE = "mixture"

    def __init__(
        self,
        p0: float,
        alpha: float,
        *,
        mode: str,
        fixed_lambda_val: float | None = None,
        epsilon: float | None = None,
        expert_qs: list[float | None] | None = None,
        prior_weights: list[float] | None = None,
    ) -> None:
        """Internal constructor — use factory classmethods."""
        self._p0 = p0
        self._alpha = alpha
        self._mode = mode
        # Fixed-lambda state
        self._fixed_lam = fixed_lambda_val
        # Mixture state — expert_qs entry is None for adaptive (cash / terminal-only)
        self._expert_qs: list[float | None] = expert_qs if expert_qs is not None else []
        self._prior_weights: list[float] = prior_weights if prior_weights is not None else []
        k = len(self._expert_qs)
        self._expert_log_wealths: list[float] = [0.0] * k
        self._expert_shares: list[float] = list(self._prior_weights)
        self._epsilon = epsilon
        # Common state
        self._log_wealth: float = 0.0
        self._max_log_wealth_ever: float = 0.0  # monotone running maximum (LLD-C §4.3)
        self._mission_count: int = 0
        self._success_sum: int = 0
        self._first_crossing_index: int | None = None

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def fixed_lambda(cls, p0: float, alpha: float, lam: float) -> GraphEProcess:
        """Create a fixed-lambda e-process — LLD-C Thm C.1 Eq (3.1)–(3.2).

        Args:
            p0: Null threshold in (0, 1).
            alpha: Significance level in (0, 1).
            lam: Betting fraction in [0, 1/p0].

        Returns:
            Initialized GraphEProcess in fixed-lambda mode.

        Raises:
            EProcessError: If any parameter is out of range.
        """
        _validate_p0_alpha(p0, alpha)
        _validate_lambda(lam, p0)
        return cls(p0=p0, alpha=alpha, mode=cls._MODE_FIXED, fixed_lambda_val=lam)

    @classmethod
    def from_sprt(cls, p0: float, alpha: float, p1: float) -> GraphEProcess:
        """Create an e-process from SPRT parameters — LLD-C §4.3 Eq (4.8).

        Sets lambda* = (p1-p0)/(p0*(1-p0)), which is the constant bet that
        recovers the Bernoulli SPRT likelihood-ratio exactly:

            1 + lambda*(y-p0) = (p1/p0)^y * ((1-p1)/(1-p0))^(1-y)

        Args:
            p0: Null threshold in (0, 1).
            alpha: Significance level in (0, 1).
            p1: Alternative probability; must satisfy p0 < p1 < 1.

        Returns:
            Initialized GraphEProcess in fixed-lambda mode with lambda*.

        Raises:
            EProcessError: If parameters are invalid or p1 <= p0.
        """
        _validate_p0_alpha(p0, alpha)
        if not (0.0 < p1 < 1.0):
            raise EProcessError(f"p1 must be in (0, 1), got {p1}")
        if p1 <= p0:
            raise EProcessError(
                f"p1 ({p1}) must be strictly greater than p0 ({p0})"
            )
        lam = (p1 - p0) / (p0 * (1.0 - p0))
        _validate_lambda(lam, p0)
        return cls(p0=p0, alpha=alpha, mode=cls._MODE_FIXED, fixed_lambda_val=lam)

    @classmethod
    def mixture(
        cls,
        p0: float,
        alpha: float,
        epsilon: float,
        extra_qs: Sequence[float] | None = None,
        prior_weights: Sequence[float] | None = None,
    ) -> GraphEProcess:
        """Create a mixture-of-experts e-process — LLD-C Thm C.3.

        Always includes:
            - Expert 0 (cash): q=p0, lambda=0 — keeps wealth > 0.
            - Expert 1 (terminal-only): Beta(1,1) Laplace-smoothed running
              average, clipped to [p0, 1-epsilon] — LLD-C Eq (5.6).

        Optional ``extra_qs`` add fixed-forecast experts with predeclared
        q values in [p0, 1-epsilon].

        Args:
            p0: Null threshold in (0, 1).
            alpha: Significance level in (0, 1).
            epsilon: Forecast-clipping constant; must satisfy
                0 < epsilon < 1-p0.
            extra_qs: Fixed-forecast expert q values in [p0, 1-epsilon].
            prior_weights: Prior weights (pi_k > 0, sum=1) for all experts
                in order: cash, terminal-only, extra... Defaults to uniform.

        Returns:
            Initialized GraphEProcess in mixture mode.

        Raises:
            EProcessError: If any parameter is out of valid range.
        """
        _validate_p0_alpha(p0, alpha)
        upper_eps = 1.0 - p0
        if not (0.0 < epsilon < upper_eps):
            raise EProcessError(
                f"epsilon must be in (0, 1-p0) = (0, {upper_eps:.4g}), got {epsilon}. "
                "LLD-C §5.1 requires epsilon < 1-p0 for the clipping interval to be non-trivial."
            )
        extra: list[float] = list(extra_qs) if extra_qs is not None else []
        for eq in extra:
            if not (p0 <= eq <= 1.0 - epsilon):
                raise EProcessError(
                    f"extra expert q={eq} must be in [p0, 1-eps] = "
                    f"[{p0}, {1.0 - epsilon:.4g}]"
                )
        # expert_qs: None = cash (k=0), None = terminal-only (k=1), then fixed extras
        expert_qs: list[float | None] = [None, None] + [float(q) for q in extra]
        k = len(expert_qs)
        if prior_weights is not None:
            pws = list(prior_weights)
            if len(pws) != k:
                raise EProcessError(
                    f"prior_weights length {len(pws)} does not match expert "
                    f"count {k} (cash + terminal-only + {len(extra)} extra)"
                )
            if any(w <= 0.0 for w in pws):
                raise EProcessError("All prior weights must be strictly positive")
            total = math.fsum(pws)
            if not math.isclose(total, 1.0, rel_tol=1e-9):
                raise EProcessError(
                    f"prior_weights must sum to 1.0, got {total}"
                )
        else:
            pws = [1.0 / k] * k
        return cls(
            p0=p0,
            alpha=alpha,
            mode=cls._MODE_MIXTURE,
            epsilon=epsilon,
            expert_qs=expert_qs,
            prior_weights=pws,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, y: int) -> EProcessUpdate:
        """Process one fully-adjudicated mission outcome — LLD-C §8.3.

        Args:
            y: Terminal graph label Y_{G,r} in {0, 1}.

        Returns:
            Immutable EProcessUpdate snapshot after this evidence index.

        Raises:
            EProcessError: If y is not 0 or 1.
        """
        if y not in (0, 1):
            raise EProcessError(f"outcome y must be 0 or 1, got {y!r}")

        if self._mode == self._MODE_FIXED:
            self._update_fixed(y)
        else:
            self._update_mixture(y)

        self._mission_count += 1
        self._success_sum += y

        # Update running maximum (monotone; never decreases) — LLD-C §4.3
        if self._log_wealth > self._max_log_wealth_ever:
            self._max_log_wealth_ever = self._log_wealth

        # Check and record first crossing — LLD-C §8.3
        threshold = -math.log(self._alpha)
        if self._first_crossing_index is None and self._log_wealth >= threshold:
            self._first_crossing_index = self._mission_count

        return EProcessUpdate(
            mission_index=self._mission_count,
            outcome=y,
            log_wealth=self._log_wealth,
            first_crossing_index=self._first_crossing_index,
        )

    def _update_fixed(self, y: int) -> None:
        """Update log wealth for fixed-lambda mode — LLD-C Eq (3.2)."""
        assert self._fixed_lam is not None
        self._log_wealth += _log_factor_fixed(self._fixed_lam, self._p0, y)

    def _get_expert_q(self, k: int) -> float:
        """Return the predictable forecast q_{k,r} for expert k — LLD-C §5.2.

        Current expert family: Expert 0 (cash, q=p0), Expert 1 (terminal-only
        Laplace-smoothed baseline), optional fixed-forecast experts (extra_qs).
        Adaptive LLD-B factor experts (LLD-C §5.2 Experts 2-3) are a documented
        power-enhancement TODO — validity is unaffected by their absence.

        Uses only F_{r-1}-measurable data (success_sum and mission_count before
        the current update step).
        """
        eps = self._epsilon
        assert eps is not None
        if k == 0:
            # Cash expert: constant at p0, lambda=0 — LLD-C Eq (5.5)
            return self._p0
        if k == 1:
            # Terminal-only: Beta(1,1) Laplace-smoothed average — LLD-C Eq (5.6)
            raw = (1.0 + self._success_sum) / (self._mission_count + 2)
            return _clip(raw, self._p0, eps)
        # Fixed-forecast expert
        q_val = self._expert_qs[k]
        assert q_val is not None
        return q_val

    def _update_mixture(self, y: int) -> None:
        """Update mixture log wealth and expert shares — LLD-C Thm C.3.

        Implements the exact aggregate-bet formula (Eq 5.19):
            E_r^mix = E_{r-1}^mix * [1 + lambda_bar_r * (y - p0)]
        where lambda_bar_r = (q_bar_r - p0) / (p0*(1-p0)) and
        q_bar_r = sum_k w_{k,r-1} * q_{k,r}.

        Expert log wealths and shares are updated using log-space arithmetic
        to avoid numerical underflow/overflow — LLD-C §8.2–8.3.
        """
        p0 = self._p0
        eps = self._epsilon
        assert eps is not None
        k_count = len(self._expert_qs)

        # Step 1: Predictable forecasts from F_{r-1} (before observing y)
        qs = [self._get_expert_q(k) for k in range(k_count)]

        # Step 2: Wealth-weighted aggregate forecast — LLD-C Eq (5.16)
        q_bar = math.fsum(self._expert_shares[k] * qs[k] for k in range(k_count))
        q_bar = _clip(q_bar, p0, eps)

        # Step 3: Per-expert log wealth update — LLD-C Eq (5.12)
        for k in range(k_count):
            self._expert_log_wealths[k] += _log_factor_lr(qs[k], p0, y)

        # Step 4: Mixture aggregate log wealth via aggregate bet — LLD-C Eq (5.19)
        self._log_wealth += _log_factor_lr(q_bar, p0, y)

        # Step 5: Recompute shares via log-sum-exp — LLD-C Eq (5.15), §8.3
        log_pi = [math.log(self._prior_weights[k]) for k in range(k_count)]
        log_unnorm = [
            log_pi[k] + self._expert_log_wealths[k] for k in range(k_count)
        ]
        log_norm = _logsumexp(log_unnorm)
        for k in range(k_count):
            self._expert_shares[k] = math.exp(log_unnorm[k] - log_norm)

    # ------------------------------------------------------------------
    # Certificate query
    # ------------------------------------------------------------------

    def certified(self, alpha: float | None = None) -> bool:
        """Return True if E_r has EVER reached 1/alpha — LLD-C Thm C.2 Eq (4.3).

        The certificate is a one-way declaration at the first crossing time
        tau_alpha (LLD-C §4.3).  Ville's inequality controls type-I error over
        the stopping time, so once certified the flag must latch: it is based on
        the running maximum of log E_r, not the current value.  A fall in wealth
        after the first crossing does NOT revoke the certificate.

        Args:
            alpha: Significance level; defaults to the stored alpha.

        Returns:
            True if the maximum log wealth ever seen meets -log(alpha).
        """
        a = alpha if alpha is not None else self._alpha
        return self._max_log_wealth_ever >= -math.log(a)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def log_wealth(self) -> float:
        """Current log e-process wealth log E_r."""
        return self._log_wealth

    @property
    def wealth(self) -> float:
        """Current e-process wealth E_r = exp(log_wealth)."""
        return math.exp(self._log_wealth)

    @property
    def mission_count(self) -> int:
        """Number of missions processed so far (evidence index r)."""
        return self._mission_count

    @property
    def max_log_wealth_ever(self) -> float:
        """Running maximum of log E_r (monotone; never decreases) — LLD-C §4.3.

        Used by certified() to implement one-way latching semantics.
        """
        return self._max_log_wealth_ever

    @property
    def first_crossing_index(self) -> int | None:
        """First r where E_r >= 1/alpha, or None if not yet certified."""
        return self._first_crossing_index

    @property
    def prior_weights(self) -> tuple[float, ...]:
        """Prior weights (pi_k) for mixture experts. Empty in fixed-lambda mode."""
        return tuple(self._prior_weights)

    @property
    def expert_log_wealths(self) -> tuple[float, ...]:
        """Per-expert log wealth E_r^(k). Empty in fixed-lambda mode."""
        return tuple(self._expert_log_wealths)

    @property
    def expert_shares(self) -> tuple[float, ...]:
        """Per-expert wealth share w_{k,r}. Empty in fixed-lambda mode."""
        return tuple(self._expert_shares)

    # ------------------------------------------------------------------
    # Mixture diagnostics
    # ------------------------------------------------------------------

    def compute_aggregate_forecast(self) -> float:
        """Return the wealth-weighted aggregate forecast for the next mission.

        Computes q_bar_r = sum_k w_{k,r-1} * q_{k,r} using current shares
        and the predictable forecasts that would be used for the next update.
        Only valid in mixture mode.

        Returns:
            Aggregate forecast in [p0, 1-epsilon].

        Raises:
            EProcessError: If called in fixed-lambda mode.

        References:
            LLD-C Eq (5.16).
        """
        if self._mode != self._MODE_MIXTURE:
            raise EProcessError("compute_aggregate_forecast is only valid in mixture mode")
        assert self._epsilon is not None
        k_count = len(self._expert_qs)
        qs = [self._get_expert_q(k) for k in range(k_count)]
        q_bar = math.fsum(self._expert_shares[k] * qs[k] for k in range(k_count))
        return _clip(q_bar, self._p0, self._epsilon)

    def terminal_only_forecast(self) -> float:
        """Return the current terminal-only expert forecast for the next mission.

        Computes clip((1 + success_sum) / (mission_count + 2), p0, 1-epsilon).
        This is the Beta(1,1)-smoothed Laplace estimate for the next mission's
        success probability, based only on completed missions.

        Returns:
            Clipped terminal-only forecast in [p0, 1-epsilon].

        Raises:
            EProcessError: If called in fixed-lambda mode.

        References:
            LLD-C Eq (5.6), §14 test #39.
        """
        if self._mode != self._MODE_MIXTURE:
            raise EProcessError("terminal_only_forecast is only valid in mixture mode")
        assert self._epsilon is not None
        raw = (1.0 + self._success_sum) / (self._mission_count + 2)
        return _clip(raw, self._p0, self._epsilon)


# ---------------------------------------------------------------------------
# Type-I calibration helper
# ---------------------------------------------------------------------------


def simulate_type1_crossing_rate(
    p0: float,
    alpha: float,
    fixed_lambda: float,
    n_streams: int = 10_000,
    n_missions: int = 200,
    rng_seed: int = 42,
) -> float:
    """Empirical type-I crossing rate under the null — LLD-C Thm C.2 Eq (4.2).

    Simulates ``n_streams`` independent Bernoulli(p0) streams and returns
    the fraction of streams for which E_r >= 1/alpha at some point in
    1..n_missions.  By Ville's inequality, this fraction must converge to
    at most alpha as n_streams -> infinity.

    Args:
        p0: Null threshold; also used as the true success probability of
            each simulated stream (boundary of the null).
        alpha: Significance level in (0, 1).
        fixed_lambda: Betting fraction in [0, 1/p0].
        n_streams: Number of independent simulation streams.
        n_missions: Maximum missions per stream.
        rng_seed: Integer seed for reproducibility.

    Returns:
        Empirical crossing rate in [0.0, 1.0].

    References:
        LLD-C Thm C.2, §14 test (Ville sharpness/type-I bound).
    """
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    threshold = -math.log(alpha)
    crossings = 0

    for _ in range(n_streams):
        ep = GraphEProcess.fixed_lambda(p0=p0, alpha=alpha, lam=fixed_lambda)
        for _m in range(n_missions):
            y = int(rng.random() < p0)
            ep.update(y)
            if ep.log_wealth >= threshold:
                crossings += 1
                break

    return crossings / n_streams
