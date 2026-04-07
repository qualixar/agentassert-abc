# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Certification module — SPRT engine and compositional guarantees.

Patent §5.5 (Composition) and §5.6 (SPRT Certification).
"""

from agentassert_abc.certification.composition import (
    pipeline_composition_bound,
    sequential_composition_bound,
)
from agentassert_abc.certification.sprt import (
    SPRTCertifier,
    SPRTDecision,
    SPRTResult,
    hoeffding_sample_size,
)

__all__ = [
    "SPRTCertifier",
    "SPRTDecision",
    "SPRTResult",
    "hoeffding_sample_size",
    "pipeline_composition_bound",
    "sequential_composition_bound",
]
