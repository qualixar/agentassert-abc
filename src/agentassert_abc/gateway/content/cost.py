# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Content operator: cost_ceiling — per-session USD spend enforcement.

Ported from the Type C content evaluator; behaviour unchanged, import paths
moved.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from agentassert_abc.process.models import DecisionResult, TypeCDecision

if TYPE_CHECKING:
    from agentassert_abc.gateway.compiler import CompiledContract
    from agentassert_abc.gateway.enforcer import SessionEnforcer
    from agentassert_abc.gateway.events import PreAction
    from agentassert_abc.gateway.violation_log import ViolationLog

logger = logging.getLogger(__name__)

# Default provider prices (USD per million tokens): (input, output).
_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "anthropic": (3.00, 15.00),
    "openai": (2.50, 10.00),
    "openrouter": (0.50, 2.00),
    "gemini": (0.075, 0.30),
}


def evaluate_cost_ceiling(
    event: PreAction,
    compiled: CompiledContract,
    accumulated_cost: float,
    violations: ViolationLog,
) -> DecisionResult | None:
    """Deny/warn/log once `accumulated_cost` reaches the configured ceiling."""
    if compiled.cost_ceiling_config is None:
        return None

    config = compiled.cost_ceiling_config
    if accumulated_cost < config.max_usd_per_session:
        return None

    reason = (
        f"Cost ceiling breached: ${accumulated_cost:.4f} >= "
        f"${config.max_usd_per_session:.2f}"
    )

    if config.action_on_breach == "deny":
        violations.record("cost_ceiling", "PreAction", event.tool, reason)
        return DecisionResult(
            decision=TypeCDecision.DENY,
            reason=reason,
            violation_name="cost_ceiling",
        )
    if config.action_on_breach == "warn":
        violations.record_soft("cost_ceiling", "PreAction", event.tool, reason)
        return None
    return None  # log: no violation entry, just pass through


def extract_usage(resp_data: dict[str, Any], provider: str) -> tuple[int, int] | None:
    """Extract (input_tokens, output_tokens) from a provider-specific response."""
    if not isinstance(resp_data, dict):
        return None

    if provider == "anthropic":
        usage = resp_data.get("usage", {})
        inp, out = usage.get("input_tokens"), usage.get("output_tokens")
        if inp is not None and out is not None:
            return (int(inp), int(out))

    elif provider in ("openai", "openrouter"):
        usage = resp_data.get("usage", {})
        inp, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if inp is not None and out is not None:
            return (int(inp), int(out))

    elif provider == "gemini":
        meta = resp_data.get("usageMetadata", {})
        inp, out = meta.get("promptTokenCount"), meta.get("candidatesTokenCount")
        if inp is not None and out is not None:
            return (int(inp), int(out))

    return None


def parse_streaming_usage(text: str) -> dict[str, Any] | None:
    """Scan SSE lines from the end for the last `data:` line carrying usage."""
    lines = text.splitlines()
    for line in reversed(lines):
        if not line.startswith("data: "):
            continue
        try:
            payload = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if "usage" in payload or "usageMetadata" in payload:
            return payload
    return None


def update_cost(
    resp_data: dict[str, Any],
    canonical: Any,  # CanonicalRequest (proxy-layer type — not ported in Phase C)
    enforcer: SessionEnforcer,
) -> None:
    """Parse usage from a response, price it, and accumulate onto `enforcer`."""
    if enforcer._compiled.cost_ceiling_config is None:
        return

    usage = extract_usage(resp_data, canonical.provider)
    if usage is None:
        return

    input_tokens, output_tokens = usage
    config = enforcer._compiled.cost_ceiling_config
    provider = canonical.provider

    # Price resolution: contract.provider_price_map > contract.global_price > default.
    if provider in config.provider_price_map:
        price_in = config.provider_price_map[provider].input
        price_out = config.provider_price_map[provider].output
    elif config.price_per_million_input is not None:
        price_in = config.price_per_million_input
        price_out = config.price_per_million_output or config.price_per_million_input
    else:
        price_in, price_out = _DEFAULT_PRICES.get(provider, (1.0, 5.0))

    cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    with enforcer._cost_lock:
        enforcer._accumulated_cost_usd += cost

    if enforcer._store is not None:
        enforcer._store.put("cost", {"accumulated_usd": enforcer._accumulated_cost_usd})
