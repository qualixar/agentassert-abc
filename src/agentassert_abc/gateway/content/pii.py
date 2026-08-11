# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Content operator: pii_filter — PII detection/redaction over agent output.

Ported from agentassert-typec's `evaluator/content_eval.py` (port-delta
§A.2, item #19). Behavior unchanged from typec; only import paths moved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentassert_abc.process.models import DecisionResult, TypeCDecision

if TYPE_CHECKING:
    from agentassert_abc.gateway.compiler import CompiledContract
    from agentassert_abc.gateway.violation_log import ViolationLog


def evaluate_pii_filter(
    text: str,
    compiled: CompiledContract,
    violations: ViolationLog,
    is_streaming: bool,
) -> DecisionResult | None:
    """Scan `text` for PII patterns and act per the contract's `pii_filter`.

    Returns None (allow / no action) if no pii_filter is configured or no
    matches are found.
    """
    if compiled.pii_filter_config is None:
        return None
    if not text:
        return None

    config = compiled.pii_filter_config
    action = config.streaming_action if is_streaming else config.action

    found: list[tuple[str, str]] = []
    for name, pattern in compiled.pii_compiled_patterns:
        for match in pattern.findall(text):
            found.append((name, str(match)[:50]))

    if not found:
        return None

    reason = f"PII detected: {', '.join(sorted({f[0] for f in found}))}"

    if action in ("log", "warn"):
        violations.record_soft("pii_filter", "PostAction", "response", reason)
        return None

    if action == "redact":
        violations.record_soft("pii_filter", "PostAction", "response", reason)
        return DecisionResult(
            decision=TypeCDecision.REDACT,
            reason=reason,
            violation_name="pii_filter",
        )

    if action == "block":
        if is_streaming:
            # Cannot block already-yielded streaming content — degrade to warn.
            violations.record_soft(
                "pii_filter", "PostAction", "response", f"[stream] {reason}"
            )
            return None
        violations.record("pii_filter", "PostAction", "response", reason)
        return DecisionResult(
            decision=TypeCDecision.DENY,
            reason=f"ContractBreach: pii_filter — {reason}",
            violation_name="pii_filter",
        )

    return None


def apply_pii_redaction(text: str, patterns: list[tuple[str, Any]]) -> str:
    """Substitute every match of `patterns` in `text` with a redaction marker."""
    result = text
    for name, pattern in patterns:
        result = pattern.sub(f"[REDACTED:{name.upper()}]", result)
    return result
