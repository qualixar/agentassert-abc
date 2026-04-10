# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Monitor result models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StepResult(BaseModel):
    """Result of a single monitoring step (one turn)."""

    model_config = ConfigDict(frozen=True)

    hard_violations: int = 0
    soft_violations: int = 0
    violated_names: list[str] = []
    violated_hard_names: list[str] = []  # M-21: Distinguish hard vs soft
    violated_soft_names: list[str] = []  # M-21: Distinguish hard vs soft
    drift_score: float = 0.0
    recovery_needed: bool = False
    recovery_attempted: bool = False
    recovery_succeeded: bool = False
    recovery_strategy: str = ""  # M-22: Which strategy to use


class SessionSummary(BaseModel):
    """Aggregated metrics for an entire session."""

    model_config = ConfigDict(frozen=True)

    turn_count: int = 0
    total_hard_violations: int = 0
    total_soft_violations: int = 0
    total_events: int = 0
    mean_c_hard: float = 1.0
    mean_c_soft: float = 1.0
    mean_drift: float = 0.0
    recovery_rate: float = 0.0
    theta: float = 1.0


class PreconditionCheckResult(BaseModel):
    """Result of checking preconditions."""

    model_config = ConfigDict(frozen=True)

    all_met: bool = True
    failed_names: list[str] = []
