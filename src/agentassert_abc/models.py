# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""ContractSpec Pydantic models — from patent TECHNICAL-ATTACHMENT.md §4.

Every model here maps directly to the ContractSpec DSL specification.
All models are frozen (immutable) and strictly typed.
Default values match the patent exactly.

Patent reference: arXiv:2602.22302, TECHNICAL-ATTACHMENT.md §4-§5
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenModel(BaseModel):
    """Base for all AgentAssert models — frozen and strict."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)


# --- Constraint Check (§4.3 — 12 operators + expr + between) ---


class ConstraintCheck(_FrozenModel):
    """Single constraint check: field + operator + value.

    Exactly one of the 14 operator fields must be set. Validated at construction
    time by the model_validator below (Ledger 3b).
    """

    field: str
    equals: Any | None = None
    not_equals: Any | None = None
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None
    in_: list[Any] | None = Field(None, alias="in")
    not_in: list[Any] | None = None
    contains: str | None = None
    not_contains: str | None = None
    matches: str | None = None
    exists: bool | None = None
    expr: str | None = None
    between: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _exactly_one_operator(self) -> ConstraintCheck:
        """Ledger 3b: exactly one of the 14 operators must be non-None."""
        ops = [
            self.equals, self.not_equals, self.gt, self.gte,
            self.lt, self.lte, self.in_, self.not_in,
            self.contains, self.not_contains, self.matches,
            self.exists, self.expr, self.between,
        ]
        non_none = sum(1 for op in ops if op is not None)
        if non_none == 0:
            raise ValueError(
                "ConstraintCheck must have exactly one operator set; none were set. "
                "Add one of: equals, not_equals, gt, gte, lt, lte, in, not_in, "
                "contains, not_contains, matches, exists, expr, between."
            )
        if non_none > 1:
            raise ValueError(
                f"ConstraintCheck must have exactly one operator set; {non_none} were set. "
                "Remove the extra operators."
            )
        return self


# --- Constraints (§4.2 — Hard/Soft separation) ---


class HardConstraint(_FrozenModel):
    """Hard invariant — must never be violated. Single violation = critical event."""

    name: str
    description: str = ""
    category: str = ""
    check: ConstraintCheck


class SoftConstraint(_FrozenModel):
    """Soft invariant — should be met but allows temporary deviation with recovery."""

    name: str
    description: str = ""
    category: str = ""
    check: ConstraintCheck
    recovery: str = ""
    recovery_window: int = Field(3, ge=1, le=1000)  # k turns (patent default)


# --- Invariants container ---


class Invariants(_FrozenModel):
    """Hard + soft constraint separation — the core of ABC."""

    hard: list[HardConstraint] = []
    soft: list[SoftConstraint] = []


# --- Governance (§4.2) ---


class GovernanceConstraint(_FrozenModel):
    """Governance constraint — operational concerns (tools, budgets, costs)."""

    name: str
    description: str = ""
    category: str = ""
    check: ConstraintCheck
    recovery: str = ""
    recovery_window: int = Field(3, ge=1, le=1000)


class Governance(_FrozenModel):
    """Governance section — hard + soft operational constraints."""

    hard: list[HardConstraint] = []
    soft: list[GovernanceConstraint] = []


# --- Recovery (§4.2 — 4 recovery types) ---


class RecoveryAction(_FrozenModel):
    """Named recovery strategy with typed actions."""

    name: str
    type: Literal[
        "inject_correction",
        "reduce_autonomy",
        "pause_and_escalate",
        "graceful_shutdown",
    ]
    actions: list[str] = []
    max_attempts: int = Field(1, ge=1, le=100)
    fallback: str | None = None


class RecoveryConfig(_FrozenModel):
    """Recovery section — named, reusable strategies."""

    strategies: list[RecoveryAction] = []


# --- Preconditions ---


class Precondition(_FrozenModel):
    """Guard clause — must hold before agent processes request."""

    name: str
    description: str = ""
    check: ConstraintCheck


# --- Metadata ---


class ContractMetadata(_FrozenModel):
    """Contract metadata for organizational management."""

    author: str = ""
    domain: str = ""
    created: str = ""
    tags: list[str] = []


# --- Satisfaction Parameters (§5.3) ---


class SatisfactionParams(_FrozenModel):
    """(p, delta, k)-satisfaction from patent §5.3.

    Defaults match the e-commerce example:
    - p=0.95: hard constraints satisfied with 95% probability
    - delta=0.1: soft deviation bound of 0.1
    - k=3: recover from soft violations within 3 turns
    """

    p: float = Field(0.95, ge=0.0, le=1.0)
    delta: float = Field(0.1, ge=0.0, le=1.0)  # δ=0 allowed (LLD-A §18-14 degenerate)
    k: int = Field(3, ge=0, le=1000)  # k=0 allowed (recover at onset turn)


# --- Drift Configuration (§5.1) ---


class DriftWeights(_FrozenModel):
    """Drift metric weights: D(t) = w_c × D_compliance + w_d × D_distributional.

    Patent defaults: compliance=0.6, distributional=0.4. Weights must sum to 1.0.
    """

    compliance: float = Field(0.6, ge=0.0, le=1.0)
    distributional: float = Field(0.4, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> DriftWeights:
        """Ledger 3a: HARD-FAIL if weights do not sum to 1.0."""
        total = self.compliance + self.distributional
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                f"DriftWeights must sum to 1.0, got {total:.6f} "
                f"(compliance={self.compliance}, distributional={self.distributional})"
            )
        return self


class DriftThresholds(_FrozenModel):
    """Drift alert thresholds. Patent defaults: warning=0.3, critical=0.6."""

    warning: float = Field(0.3, ge=0.0, le=1.0)
    critical: float = Field(0.6, ge=0.0, le=1.0)


class DriftConfig(_FrozenModel):
    """Complete drift configuration from patent §5.1."""

    weights: DriftWeights = DriftWeights()
    window: int = Field(50, ge=1, le=10000)
    thresholds: DriftThresholds = DriftThresholds()


# --- Reliability Configuration (§5.7) ---


class ReliabilityWeights(_FrozenModel):
    """Reliability Index weights: Θ = 0.35C̄ + 0.25(1-D̄) + 0.20(1/(1+E)) + 0.20S.

    Patent §5.7 names:
    - compliance (0.35): How well the agent follows constraints
    - drift (0.25): How stable behavior is over time
    - event_freq (0.20): How rarely violations occur — 1/(1+E)
    - recovery_success (0.20): How effectively the agent recovers — S

    M-16: Aliases 'recovery' and 'stress' accepted for backward compat
    with patent YAML example which uses those names. Weights must sum to 1.0.
    """

    compliance: float = Field(0.35, ge=0.0, le=1.0)
    drift: float = Field(0.25, ge=0.0, le=1.0)
    event_freq: float = Field(0.20, ge=0.0, le=1.0, alias="stress")
    recovery_success: float = Field(0.20, ge=0.0, le=1.0, alias="recovery")

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ReliabilityWeights:
        """Ledger 3a: HARD-FAIL if weights do not sum to 1.0."""
        total = self.compliance + self.drift + self.event_freq + self.recovery_success
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                f"ReliabilityWeights must sum to 1.0, got {total:.6f} "
                f"(compliance={self.compliance}, drift={self.drift}, "
                f"event_freq={self.event_freq}, recovery_success={self.recovery_success})"
            )
        return self


class ReliabilityConfig(_FrozenModel):
    """Complete reliability configuration from patent §5.7."""

    weights: ReliabilityWeights = ReliabilityWeights()
    deployment_threshold: float = Field(0.90, ge=0.0, le=1.0)


# --- Root ContractSpec (§4.1) ---


class ContractSpec(_FrozenModel):
    """Root model — the entire ContractSpec YAML contract.

    This is the primary data structure that represents a behavioral contract.
    Corresponds to TECHNICAL-ATTACHMENT.md §4.1.
    """

    contractspec: str
    kind: Literal["agent", "pipeline"]
    name: str
    description: str
    version: str
    metadata: ContractMetadata | None = None
    preconditions: list[Precondition] = []
    invariants: Invariants | None = None
    governance: Governance | None = None
    recovery: RecoveryConfig | None = None
    satisfaction: SatisfactionParams | None = None
    drift: DriftConfig | None = None
    reliability: ReliabilityConfig | None = None
