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
# Meta Contributor API model identifier (Meta Model API /v1/responses).
# The Spark family are reasoning models; META_REASONING_EFFORT below keeps them
# inside the FRONTIER_MAX_OUTPUT_TOKENS cap.
META_CONTRIBUTOR_MODEL: Final[str] = "muse-spark-1.2-contributor"

# Reasoning effort for Meta Spark (OpenAI Responses `reasoning.effort`).
# "minimal" completes a short answer in ~85 reasoning tokens (verified), so the
# visible answer fits within FRONTIER_MAX_OUTPUT_TOKENS=160.  "low"/"medium"/
# "high" overrun the cap; "none" is rejected by the model.  Part of frozen
# sampling (LLD-E §5.1) — record in the preregistration.
META_REASONING_EFFORT: Final[str] = "minimal"

# Anchor model offered through OpenRouter — robust, cheap, NON-reasoning
# instruct model.  Verified correct on capability probes; a reasoning model
# here returns null content under the 160-token cap (rejected loudly).
OPENROUTER_DEFAULT_MODEL: Final[str] = "mistralai/mistral-small-24b-instruct-2501"

# GrokBridgeClient local bridge proxy base URL.  Override via env var
# GROK_PROXY_BASE_URL before constructing the adapter.
GROK_PROXY_BASE_URL: Final[str] = "http://localhost:8787/v1"

# Provider price table (input_per_M_USD, output_per_M_USD).
# Used by provider adapters in providers.py to compute cost_usd per response.
# LLD-E §6.2 admission ceilings still apply; any model priced above those
# ceilings must not be admitted.
#   meta_contributor   : Meta Contributor API — $0.10 / $0.20 per 1M tokens
#   openrouter_default : OpenRouter fallback   — $0.05 / $0.15 per 1M tokens
#                        (conservative UPPER bound across the OpenRouter roster;
#                        the authoritative per-call cost is OpenRouter's own
#                        usage.cost field when present — see providers.py)
#   grok_bridge        : local subscription-backed proxy — $0.00 / $0.00
#
# To wire the §6.3 batch gate when using a frontier adapter with
# _execute_mission_batch (run.py), pass:
#   per_call_ceiling=config.PER_CALL_CEILING_USD
# This is NOT automatic; the caller must supply it.  Since FRONTIER_ENABLED is
# False by default, frontier adapters are inert until explicitly enabled.
PROVIDER_PRICES: Final[dict[str, tuple[float, float]]] = {
    "meta_contributor": (0.10, 0.20),
    "openrouter_default": (0.05, 0.15),
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
# All IDs verified present on OpenRouter and NON-reasoning (clean content under
# the 160-token output cap).  The roster is a nested design anchored on
# OPENROUTER_DEFAULT_MODEL (mistral-small-24b): each condition flips exactly one
# factor — same model, then same vendor / different model, then different vendor.
#
# same_vendor pair member: a smaller Mistral model (same vendor as the anchor,
#   different size / checkpoint).  Verified correct on capability probes.
OPENROUTER_SAME_VENDOR_MODEL: Final[str] = "mistralai/ministral-8b-2512"
#
# different_vendor pair member: Gemma 3 12B (Google) via OpenRouter — different
#   vendor from the Mistral anchor, cheap, within the LLD-E §6.2 ceiling.
#   Verified correct on capability probes.
OPENROUTER_DIFF_VENDOR_MODEL: Final[str] = "google/gemma-3-12b-it"
#
# Grok model ID for the GrokBridgeClient breadth arm (local bridge proxy,
#   subscription-backed, $0 per-call).
GROK_MODEL: Final[str] = "grok-4.5"

# --- Domain missions + robust runner (LLD-F) --------------------------------
# Domains enabled for the confirmatory frontier run (retail + financial).
EXPERIMENT_DOMAINS: Final[tuple[str, ...]] = ("retail", "financial")

# Maximum total call attempts per frontier request (1 attempt + N-1 retries).
# With FRONTIER_MAX_RETRIES=4: one initial attempt + up to 3 exponential-
# backoff retries.  Default is 4 (total attempts), not 4 retries.
FRONTIER_MAX_RETRIES: Final[int] = 4

# Base delay in seconds for exponential backoff between retry attempts.
# Actual delay = FRONTIER_BACKOFF_BASE_S * 2**attempt + uniform(0, 0.1).
FRONTIER_BACKOFF_BASE_S: Final[float] = 1.0

# Maximum concurrent worker threads for frontier mission execution (LLD-F §G.1).
# Frontier tier passes this value to _execute_mission_batch; dry/local stays at 1
# (serial path — byte-identical to the pre-concurrency runner).
FRONTIER_CONCURRENCY: Final[int] = 16
