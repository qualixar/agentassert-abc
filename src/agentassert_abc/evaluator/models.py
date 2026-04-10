# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Evaluator result models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ConstraintResult(BaseModel):
    """Result of evaluating a single constraint."""

    model_config = ConfigDict(frozen=True)

    name: str
    satisfied: bool
    evidence: str
    constraint_type: Literal["hard", "soft", "governance", "precondition"]


class EvaluationResult(BaseModel):
    """Result of evaluating all constraints in a contract against a state."""

    model_config = ConfigDict(frozen=True)

    hard_results: list[ConstraintResult] = []
    soft_results: list[ConstraintResult] = []
    governance_results: list[ConstraintResult] = []
    c_hard: float = 1.0
    c_soft: float = 1.0
    hard_violations: list[ConstraintResult] = []
    soft_violations: list[ConstraintResult] = []
