# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Shared configuration for the $20-capped validation experiment (LLD-E).

Single source of truth so every harness component agrees on budget, models,
thresholds, and safety flags.

SAFETY: no real paid API call may happen unless :data:`FRONTIER_ENABLED` is
explicitly set True *and* the :class:`~agentassert_abc.experiments.budget`
ledger permits the spend. Default is off.
"""
from __future__ import annotations

from typing import Final

# --- Safety gates -----------------------------------------------------------
FRONTIER_ENABLED: Final[bool] = False  # flip on ONLY with explicit approval
BUDGET_CAP_USD: Final[float] = 20.0
BUDGET_STOP_USD: Final[float] = 19.50  # hard stop, $0.50 reporting-lag buffer (LLD-E §6)

# --- Frontier per-call caps + admission price ceilings (LLD-E §6.2) ---------
FRONTIER_MAX_INPUT_TOKENS: Final[int] = 800
FRONTIER_MAX_OUTPUT_TOKENS: Final[int] = 160
MAX_INPUT_PRICE_PER_M_USD: Final[float] = 5.0
MAX_OUTPUT_PRICE_PER_M_USD: Final[float] = 20.0
# Conservative worst-case cost of one admitted frontier call = $0.0072
PER_CALL_CEILING_USD: Final[float] = (
    FRONTIER_MAX_INPUT_TOKENS / 1e6 * MAX_INPUT_PRICE_PER_M_USD
    + FRONTIER_MAX_OUTPUT_TOKENS / 1e6 * MAX_OUTPUT_PRICE_PER_M_USD
)

# --- Local model tier (free, Ollama — see ~/.claude/rules/local-ai.md) ------
OLLAMA_URL: Final[str] = "http://localhost:11434"
LOCAL_MODELS: Final[tuple[str, ...]] = ("qwen2.5:7b", "gemma3:4b", "llama3.2")

# --- Certification / statistics defaults (LLD-C, LLD-B, LLD-E) --------------
P0_RELIABILITY: Final[float] = 0.90  # e-process null threshold
ALPHA: Final[float] = 0.05
TAU_EPS: Final[float] = 0.05
LOCAL_N_PER_CONDITION: Final[int] = 6000
FRONTIER_N_PER_CONDITION: Final[int] = 120
