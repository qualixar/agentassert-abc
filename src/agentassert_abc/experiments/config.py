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

# --- Provider adapter configuration (LLD-E §4.1, Task #20) ------------------
# Meta Contributor API model identifier.
META_CONTRIBUTOR_MODEL: Final[str] = "meta-llama/llama-4-scout-17b-16e-instruct"

# Default model offered through OpenRouter for the Qwen3.7-Flash price tier.
OPENROUTER_DEFAULT_MODEL: Final[str] = "qwen/qwen3-7b-fast"

# GrokBridgeClient local hermes proxy base URL.  Override via env var
# GROK_PROXY_BASE_URL before constructing the adapter.
GROK_PROXY_BASE_URL: Final[str] = "http://localhost:8787/v1"

# Provider price table (input_per_M_USD, output_per_M_USD).
# Used by provider adapters in providers.py to compute cost_usd per response.
# LLD-E §6.2 admission ceilings still apply; any model priced above those
# ceilings must not be admitted.
#   meta_contributor      : Meta Contributor API   — $0.10 / $0.20 per 1M tokens
#   openrouter_qwen_flash : OpenRouter Qwen3.7-F   — $0.03 / $0.13 per 1M tokens
#   grok_bridge           : local subscription-backed proxy — $0.00 / $0.00
#
# To wire the §6.3 batch gate when using a frontier adapter with
# _execute_mission_batch (run.py), pass:
#   per_call_ceiling=config.PER_CALL_CEILING_USD
# This is NOT automatic; the caller must supply it.  Since FRONTIER_ENABLED is
# False by default, frontier adapters are inert until explicitly enabled.
PROVIDER_PRICES: Final[dict[str, tuple[float, float]]] = {
    "meta_contributor": (0.10, 0.20),
    "openrouter_qwen_flash": (0.03, 0.13),
    "grok_bridge": (0.0, 0.0),
}

# --- Certification / statistics defaults (LLD-C, LLD-B, LLD-E) --------------
P0_RELIABILITY: Final[float] = 0.90  # e-process null threshold
ALPHA: Final[float] = 0.05
TAU_EPS: Final[float] = 0.05
LOCAL_N_PER_CONDITION: Final[int] = 6000
FRONTIER_N_PER_CONDITION: Final[int] = 120

# --- Frontier model roster constants (LLD-E §4.1, Task #20 extension) --------
# Locked before the first confirmatory run; substitution only via a dated
# preregistration amendment BEFORE any confirmatory outcome is generated.
#
# same_vendor pair: two Alibaba Qwen models via OpenRouter (different sizes,
#   same family / vendor / runtime).
OPENROUTER_QWEN_SAME_VENDOR: Final[str] = "qwen/qwen2.5-3b-instruct"
#
# different_vendor pair: Gemma 3 1B (Google) via OpenRouter — minimal cost
#   within the LLD-E §6.2 admission ceiling.
OPENROUTER_GEMMA_DIFF_VENDOR: Final[str] = "google/gemma-3-1b-it"
#
# Grok model ID for the GrokBridgeClient breadth arm (local hermes proxy,
#   subscription-backed, $0 per-call).
GROK_MODEL: Final[str] = "grok-3-mini"
