# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""F2 (p, δ, k)-Satisfaction session-level check — LLD-A v2 §3.4, §16.

An agent satisfies a contract with parameters (p, δ, k) iff:

1. **Hard** — ``Pr[C_hard(t) = 1 for all t] ≥ p`` (session hard guarantee,
   LLD-A §17 Def 3.7).
2. **Soft (threshold-and-recovery)** — every soft *excursion onset* recovers
   within ``k`` turns.  δ defines the acceptable-region boundary ``1 − δ``:
   turn ``t`` is *below* when ``C_soft(t) < 1 − δ``.  An **onset** is a turn
   that crosses into the below region (below at ``t``, not below at ``t − 1``).
   The obligation from an onset at ``t0`` is **recovered** iff
   ``∃ u ∈ {t0, …, t0 + k} : C_soft(u) ≥ 1 − δ``.

Key v2 semantics (LLD-A §3.4):

* A soft excursion is **not** an immediate breach — it becomes a breach only
  when its recovery deadline (``t0 + k``) expires unrecovered.
* An uninterrupted below-threshold episode is a **single** obligation (one
  onset); one recovery discharges it.
* An onset whose window ``[t0, t0 + k]`` extends past the last observed turn
  without recovery is **PENDING**, not failed (LLD-A §15, §18).  The verdict
  exposes both §18 bounds: :attr:`~SatisfactionVerdict.passed` (pessimistic —
  pending counts as breach) and
  :attr:`~SatisfactionVerdict.passed_optimistic` (pending counts as recovered).

The v1 patent-attachment reading — a deterministic ``max_t|C_soft − 1| ≤ δ``
deviation cap plus recovery to *exactly* 1 — is **SUPERSEDED** (LLD-A §16, §17)
and is deliberately NOT implemented here.  ``max_soft_deviation`` is retained
as an informational metric only; it is never a pass/fail condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentassert_abc.models import SatisfactionParams

_EPS = 1e-9  # float tolerance for boundary comparisons


@dataclass(frozen=True)
class TurnRecord:
    """Single turn's compliance data within a session.

    Attributes:
        c_hard: Fraction of hard constraints satisfied (1.0 = all met).
        c_soft: Fraction of soft constraints satisfied (1.0 = all met).
        recovery_event: Whether a recovery was applied this turn (informational).
    """

    c_hard: float = 1.0
    c_soft: float = 1.0
    recovery_event: bool = False


@dataclass(frozen=True)
class SessionLog:
    """Immutable record of a session's per-turn compliance.

    Attributes:
        turns: Ordered tuple of :class:`TurnRecord`, one per interaction turn.
    """

    turns: tuple[TurnRecord, ...] = ()

    def __len__(self) -> int:
        return len(self.turns)


@dataclass(frozen=True)
class SatisfactionVerdict:
    """Result of a (p, δ, k)-satisfaction check (LLD-A v2 §3.4, §16, §18).

    Attributes:
        passed: Pessimistic (§18 lower-bound) verdict — hard guarantee holds
            AND there is no soft breach AND no pending excursion.  This is the
            conservative default for a reliability decision.
        passed_optimistic: Optimistic (§18 upper-bound) verdict — hard holds
            AND no *definite* breach; pending excursions are assumed to recover.
        p_observed: Observed all-hard-compliance probability (1.0/0.0 for a
            single session; aggregated across sessions by the SPRT/e-process).
        soft_breaches: Number of excursion onsets whose full window
            ``[t0, t0+k]`` was observed with no recovery — definite failures.
        soft_pending: Number of onsets whose window runs past the observed
            horizon without recovery yet — undecided (LLD-A §15).
        max_recovery_window: Largest observed onset→recovery lag (turns).
        max_soft_deviation: ``max_t |C_soft(t) − 1|`` — informational only;
            NOT a v2 pass condition (the v1 deviation cap is superseded).
        failure_reasons: Human-readable reasons contributing to a non-pass.
    """

    passed: bool
    p_observed: float
    max_soft_deviation: float
    max_recovery_window: int
    soft_breaches: int = 0
    soft_pending: int = 0
    passed_optimistic: bool = False
    failure_reasons: tuple[str, ...] = ()


class SatisfactionChecker:
    """Checks whether a session log satisfies F2 (p, δ, k) — LLD-A v2 semantics.

    Usage::

        checker = SatisfactionChecker(params)
        verdict = checker.check_session(session_log)
        if verdict.passed:            # pessimistic / §18 lower bound
            ...
    """

    def __init__(self, params: SatisfactionParams) -> None:
        self._params = params

    def check_session(self, session_log: SessionLog) -> SatisfactionVerdict:
        """Evaluate the hard guarantee and the soft excursion-recovery rule.

        Args:
            session_log: The session's per-turn compliance record.

        Returns:
            :class:`SatisfactionVerdict` with pessimistic + optimistic verdicts,
            breach/pending counts, and detailed metrics.
        """
        turns = session_log.turns
        n = len(turns)
        reasons: list[str] = []
        delta = self._params.delta
        k = self._params.k
        threshold = 1.0 - delta  # acceptable region: c_soft >= threshold

        def _acceptable(idx: int) -> bool:
            return turns[idx].c_soft >= threshold - _EPS

        def _below(idx: int) -> bool:
            return 0 <= idx < n and not _acceptable(idx)

        # --- Condition 1: Pr[C_hard(t) = 1 for all t] >= p (LLD-A §17 Def 3.7)
        all_hard = all(t.c_hard >= 1.0 - _EPS for t in turns)
        p_observed = 1.0 if all_hard else 0.0
        cond1 = p_observed >= self._params.p
        if not cond1:
            n_hard_viol = sum(1 for t in turns if t.c_hard < 1.0 - _EPS)
            reasons.append(
                f"F2/C1: hard p_observed={p_observed:.2f} < required "
                f"p={self._params.p} ({n_hard_viol} hard-violation turn(s))"
            )

        # --- Condition 2 (v2): soft excursion-onset recovery (LLD-A §3.4) ---
        soft_breaches = 0
        soft_pending = 0
        max_window = 0
        for t0 in range(n):
            # Onset: below at t0, not below at t0-1 (D^S(-1)=0).
            if not (_below(t0) and not _below(t0 - 1)):
                continue
            # Recovery witness u in {t0, ..., t0+k} with c_soft(u) >= 1-δ.
            last_u = min(t0 + k, n - 1)
            recovered_at: int | None = None
            for u in range(t0, last_u + 1):
                if _acceptable(u):
                    recovered_at = u
                    break
            if recovered_at is not None:
                max_window = max(max_window, recovered_at - t0)
            elif t0 + k <= n - 1:
                # Full window observed, no recovery -> definite breach.
                soft_breaches += 1
                reasons.append(
                    f"F2/C2: soft excursion onset at turn {t0} not recovered "
                    f"within k={k} (window [{t0}, {t0 + k}])"
                )
            else:
                # Window extends past observed horizon -> pending (LLD-A §15).
                soft_pending += 1

        if soft_pending and soft_breaches == 0:
            reasons.append(
                f"F2/C2: {soft_pending} soft excursion(s) PENDING — recovery "
                "window extends past the observed horizon (LLD-A §15/§18)"
            )

        max_dev = max((abs(1.0 - t.c_soft) for t in turns), default=0.0)
        passed = cond1 and soft_breaches == 0 and soft_pending == 0
        passed_optimistic = cond1 and soft_breaches == 0
        return SatisfactionVerdict(
            passed=passed,
            p_observed=p_observed,
            max_soft_deviation=max_dev,
            max_recovery_window=max_window,
            soft_breaches=soft_breaches,
            soft_pending=soft_pending,
            passed_optimistic=passed_optimistic,
            failure_reasons=tuple(reasons),
        )
