# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Sequential Probability Ratio Test (SPRT) Certification Engine — Patent §5.6.

Implements Wald's SPRT for agent compliance certification:
- H0: p <= p0 (agent does NOT meet compliance threshold)
- H1: p >= p1 (agent DOES meet compliance threshold)

The SPRT tests sessions sequentially and stops when evidence is
sufficient, achieving 50-80% efficiency over fixed-sample (Hoeffding).

Expected stopping time: E[N*] ~= log(1/alpha) / D_KL(p_hat || p0)
Hoeffding comparison:   N_H  = log(2/alpha) / (2 * epsilon^2)

Patent reference: arXiv:2602.22302, TECHNICAL-ATTACHMENT.md §5.6
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SPRTDecision(Enum):
    """Terminal decision from the SPRT certifier."""

    CONTINUE = "continue"
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True)
class SPRTResult:
    """Result of an SPRT update step.

    Attributes:
        decision: Current decision — CONTINUE, ACCEPT, or REJECT.
        sessions_used: Total sessions processed so far.
        log_likelihood_ratio: Cumulative log-likelihood ratio (Lambda_n).
    """

    decision: SPRTDecision
    sessions_used: int
    log_likelihood_ratio: float


class SPRTCertifier:
    """Sequential Probability Ratio Test for agent compliance.

    Wald's SPRT with boundaries:
        A = log((1 - beta) / alpha)   — accept H1 when Lambda >= A
        B = log(beta / (1 - alpha))   — reject H1 (accept H0) when Lambda <= B

    Each session updates:
        Lambda_n += log(p1/p0) if passed, log((1-p1)/(1-p0)) if failed.
    """

    def __init__(self, p0: float, p1: float, alpha: float, beta: float) -> None:
        """Initialize the SPRT certifier.

        Args:
            p0: Null hypothesis threshold (agent does NOT meet this).
            p1: Alternative hypothesis threshold (agent DOES meet this).
            alpha: Type I error rate (false accept). Must be in (0, 1).
            beta: Type II error rate (false reject). Must be in (0, 1).

        Raises:
            ValueError: If parameters are out of valid range.
        """
        if not (0.0 < alpha < 1.0):
            msg = f"alpha must be in (0, 1), got {alpha}"
            raise ValueError(msg)
        if not (0.0 < beta < 1.0):
            msg = f"beta must be in (0, 1), got {beta}"
            raise ValueError(msg)
        if not (0.0 < p0 < 1.0):
            msg = f"p0 must be in (0, 1), got {p0}"
            raise ValueError(msg)
        if not (0.0 < p1 < 1.0):
            msg = f"p1 must be in (0, 1), got {p1}"
            raise ValueError(msg)
        if p0 >= p1:
            msg = f"p0 ({p0}) must be strictly less than p1 ({p1})"
            raise ValueError(msg)

        self._p0 = p0
        self._p1 = p1
        self._alpha = alpha
        self._beta = beta

        # Wald boundaries (log scale)
        self._upper_bound = math.log((1.0 - beta) / alpha)  # Accept H1
        self._lower_bound = math.log(beta / (1.0 - alpha))  # Reject H1

        # Incremental log-likelihood ratios
        self._log_lr_pass = math.log(p1 / p0)
        self._log_lr_fail = math.log((1.0 - p1) / (1.0 - p0))

        # Mutable state
        self._lambda_n: float = 0.0
        self._sessions: int = 0
        self._terminal_result: SPRTResult | None = None

    @property
    def p0(self) -> float:
        """Null hypothesis threshold."""
        return self._p0

    @property
    def p1(self) -> float:
        """Alternative hypothesis threshold."""
        return self._p1

    def update(self, session_passed: bool) -> SPRTResult:
        """Process one session result and return the current decision.

        After a terminal decision (ACCEPT or REJECT), subsequent calls
        return the same terminal result without changing state.

        Args:
            session_passed: Whether the agent passed this session.

        Returns:
            SPRTResult with current decision, sessions used, and LLR.
        """
        # Idempotent after terminal decision
        if self._terminal_result is not None:
            return self._terminal_result

        self._sessions += 1
        if session_passed:
            self._lambda_n += self._log_lr_pass
        else:
            self._lambda_n += self._log_lr_fail

        # Check boundaries
        decision = SPRTDecision.CONTINUE
        if self._lambda_n >= self._upper_bound:
            decision = SPRTDecision.ACCEPT
        elif self._lambda_n <= self._lower_bound:
            decision = SPRTDecision.REJECT

        result = SPRTResult(
            decision=decision,
            sessions_used=self._sessions,
            log_likelihood_ratio=self._lambda_n,
        )

        if decision != SPRTDecision.CONTINUE:
            self._terminal_result = result

        return result


def hoeffding_sample_size(alpha: float, epsilon: float) -> int:
    """Compute fixed-sample size via Hoeffding bound.

    N_H = ceil(log(2/alpha) / (2 * epsilon^2))

    This is the baseline that SPRT improves upon (50-80% fewer samples).

    Args:
        alpha: Significance level. Must be in (0, 1).
        epsilon: Tolerance (half-width of confidence interval). Must be > 0.

    Returns:
        Required number of samples (integer, ceiling).

    Raises:
        ValueError: If parameters are out of valid range.
    """
    if not (0.0 < alpha < 1.0):
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)
    if epsilon <= 0.0:
        msg = f"epsilon must be positive, got {epsilon}"
        raise ValueError(msg)

    return math.ceil(math.log(2.0 / alpha) / (2.0 * epsilon**2))
