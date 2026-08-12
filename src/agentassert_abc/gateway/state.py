# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Flattening agent output into constraint-evaluation state.

The evaluator looks fields up as **flat dotted keys** — ``state["output.safe"]``,
never ``state["output"]["safe"]`` (see ``evaluator/operators.py``, H-16). Every
integration is responsible for flattening its own framework output into that
shape via ``extract_state()``.

Two integrations shipped without doing so. The HTTP proxy passed
``state={"response_bytes": N}`` and the Claude Code hook passed no state at all,
while putting the real payload in ``result``. Because a missing field evaluates
to ``False``, every semantic invariant scored 0 and logged a violation — a
*compliant* agent was reported as failing, with Θ suppressed by roughly half a
point per session. This module supplies the missing flattening, plus a
load-time gate so an unsatisfiable contract fails loudly instead of silently
producing confident nonsense.
"""

from __future__ import annotations

from typing import Any

from agentassert_abc.exceptions import ContractLoadError

# Anything under `output.` is derived from the response body, so whether a given
# key is present is a per-response fact, not a contract-load fact — always allowed.
RESPONSE_NAMESPACE = "output."

# Fields each surface populates on EVERY turn, regardless of response shape.
# A constraint outside `output.` and outside its surface's set can never be
# satisfied there, so the contract is rejected at load rather than scoring the
# agent as violating forever.
PROXY_PROVIDED_FIELDS: frozenset[str] = frozenset(
    {
        "latency_ms",
        "response.bytes",
        "response.latency_ms",
        "response.status",
        "response.streamed",
        "tool.name",
    }
)
HOOK_PROVIDED_FIELDS: frozenset[str] = frozenset({"tool.name"})
# The MCP guard sees the tool call and its result, and additionally knows which
# downstream server it is guarding — so a contract can scope an invariant to one
# server when several are guarded behind the same client.
MCP_PROVIDED_FIELDS: frozenset[str] = frozenset({"tool.name", "tool.server"})

_MAX_FLATTEN_DEPTH = 6


def flatten_output(payload: Any, prefix: str = "output") -> dict[str, Any]:
    """Flatten framework output into flat dotted keys under ``prefix``.

    Follows the convention the framework adapters already use: mappings expand
    key by key (``output.decision``), nested mappings join with dots
    (``output.usage.tokens``), and anything scalar or unstructured lands on
    ``<prefix>.raw`` so text-matching constraints still have something to read.

    Args:
        payload: decoded response body, tool output, or plain text.
        prefix: namespace for the flattened keys.

    Returns:
        A flat dict. Never raises — an integration must not fail a request
        because a provider returned an unexpected shape.
    """
    state: dict[str, Any] = {}
    if payload is None:
        return state
    if isinstance(payload, str | int | float | bool):
        state[f"{prefix}.raw"] = payload
        return state
    if isinstance(payload, dict):
        _flatten_mapping(payload, prefix, state, depth=0)
        return state
    if isinstance(payload, (list, tuple)):
        state[f"{prefix}.raw"] = payload
        state[f"{prefix}.length"] = len(payload)
        return state
    # Pydantic models and similar structured objects
    dump = getattr(payload, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
        except Exception:  # noqa: BLE001 - never fail a request on output shape
            dumped = None
        if isinstance(dumped, dict):
            _flatten_mapping(dumped, prefix, state, depth=0)
            return state
    state[f"{prefix}.raw"] = str(payload)
    return state


def flatten_state(mapping: dict[str, Any]) -> dict[str, Any]:
    """Flatten a state dict in place of its own namespace, adding dotted paths.

    Unlike :func:`flatten_output` this adds **no prefix**: top-level keys keep
    the names the caller gave them, and nested mappings gain dotted aliases.
    ``{"output": {"safe": True}}`` becomes
    ``{"output": {...}, "output.safe": True}``.

    This is what lets one contract behave identically on both planes. The
    enforcement plane flattens tool output into ``output.*``; a framework
    adapter that passed a nested dict straight through produced
    ``state["output"]["safe"]``, which the evaluator never looks up (it reads
    literal dotted keys, H-16), so the invariant scored ``False`` forever. The
    transformation is purely additive — every original key survives — so a
    contract written against already-flat state is unaffected.

    Args:
        mapping: the caller's state dict.

    Returns:
        A new dict with the original entries plus a dotted alias for every
        nested value.
    """
    out: dict[str, Any] = dict(mapping)
    for key, value in mapping.items():
        if isinstance(value, dict):
            _flatten_mapping(value, str(key), out, depth=0)
    return out


def _flatten_mapping(
    mapping: dict[Any, Any], prefix: str, out: dict[str, Any], depth: int
) -> None:
    if depth >= _MAX_FLATTEN_DEPTH:
        return
    for key, value in mapping.items():
        dotted = f"{prefix}.{key}"
        if isinstance(value, dict):
            # Record the container too, so `exists` checks on it still work.
            out[dotted] = value
            _flatten_mapping(value, dotted, out, depth + 1)
        else:
            out[dotted] = value


def contract_field_names(spec: Any) -> tuple[str, ...]:
    """Every constraint field a contract's hard and soft invariants reference."""
    fields: list[str] = []
    invariants = getattr(spec, "invariants", None)
    if invariants is None:
        return ()
    for group in ("hard", "soft"):
        for constraint in getattr(invariants, group, None) or ():
            check = getattr(constraint, "check", None)
            field = getattr(check, "field", None)
            if field:
                fields.append(str(field))
    return tuple(dict.fromkeys(fields))


def assert_evaluable_on_response_surface(
    spec: Any, surface: str, provided: frozenset[str] = PROXY_PROVIDED_FIELDS
) -> None:
    """Reject a contract whose invariants can never be evaluated on this surface.

    A response-level integration only ever sees the provider response and the
    tool call. It can therefore satisfy anything under :data:`RESPONSE_NAMESPACE`
    (derived from the response body) plus the fixed fields it always sets. A
    constraint outside both would evaluate ``False`` on every turn forever — an
    agent marked non-compliant for a reason it cannot influence.

    Args:
        spec: the loaded contract.
        surface: human name for the integration, used in the error.
        provided: fields this surface populates on every turn.

    Raises:
        ContractLoadError: naming the offending fields and the surface.
    """
    unusable = [
        field
        for field in contract_field_names(spec)
        if not field.startswith(RESPONSE_NAMESPACE) and field not in provided
    ]
    if not unusable:
        return
    allowed = ", ".join(sorted(provided))
    raise ContractLoadError(
        f"contract '{getattr(spec, 'name', '?')}' cannot be enforced by the "
        f"{surface}: constraint field(s) {', '.join(sorted(unusable))} are neither "
        f"under '{RESPONSE_NAMESPACE}' nor among the fields this surface always "
        f"supplies ({allowed}). The {surface} only observes the provider response "
        "and the tool call, so these would score as violations on every turn "
        "regardless of agent behaviour. Re-express them over the response, or use "
        "a framework integration that can supply the state."
    )
