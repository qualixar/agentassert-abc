# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Regression tests for LLD-audit CRIT findings (Type C consolidation).

Locks in three fixes the independent enforcement-kernel audit surfaced:
- CRIT-2: PostAction-detected hard violations must feed the reliability event
  term (E) and the ViolationLog — previously silently dropped (Θ inflated).
- CRIT-3: accumulated Θ penalty must be observable via SessionEnd.
- CRIT-1: distributional drift is opt-in via set_drift_reference() (inert until
  calibrated, by design) — the pass-through must actually activate the JSD term.
"""

from __future__ import annotations

from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.gateway.events import PostAction
from agentassert_abc.process.models import ContractSpecExtended


def _hard_safe_contract() -> ContractSpecExtended:
    return ContractSpecExtended(
        contractspec="1.0",
        kind="agent",
        name="t",
        description="d",
        version="1.0",
        invariants={
            "hard": [{"name": "must-be-safe", "check": {"field": "safe", "equals": True}}]
        },
    )


def test_post_action_hard_violation_counts_toward_events_and_log() -> None:
    """CRIT-2: a violating PostAction increments E and records to the log."""
    enf = SessionEnforcer(_hard_safe_contract())

    enf.evaluate(PostAction(session_id="s", contract_id="t", tool="x", state={"safe": False}))

    assert enf._theta._violation_count == 1
    assert enf._violations.hard_count() == 1


def test_post_action_compliant_does_not_count() -> None:
    """CRIT-2: a compliant PostAction must not inflate E or the log."""
    enf = SessionEnforcer(_hard_safe_contract())

    enf.evaluate(PostAction(session_id="s", contract_id="t", tool="y", state={"safe": True}))

    assert enf._theta._violation_count == 0
    assert enf._violations.hard_count() == 0


def test_session_end_surfaces_theta_penalty() -> None:
    """CRIT-3: SessionEnd exposes the accumulated Θ penalty."""
    enf = SessionEnforcer(_hard_safe_contract())
    enf._theta.apply_penalty(0.03)

    end = enf.close()

    assert abs(end.theta_penalty - 0.03) < 1e-9


def test_drift_reference_activates_distributional_term() -> None:
    """CRIT-1: without calibration the JSD term is inert; set_drift_reference
    activates it, so a distribution-shifted turn produces non-zero drift even
    when compliance is perfect."""
    # Uncalibrated: perfect compliance + tool shift -> drift stays 0.
    enf_uncal = SessionEnforcer(_hard_safe_contract())
    enf_uncal.evaluate(PostAction(session_id="s", contract_id="t", tool="b", state={"safe": True}))
    assert enf_uncal._drift.history[-1] == 0.0

    # Calibrated against a different reference distribution -> JSD term fires.
    enf_cal = SessionEnforcer(_hard_safe_contract())
    enf_cal.set_drift_reference({"a": 1.0})
    enf_cal.evaluate(PostAction(session_id="s", contract_id="t", tool="b", state={"safe": True}))
    assert enf_cal._drift.history[-1] > 0.0
