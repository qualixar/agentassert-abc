# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Process / enforcement-plane models (Type C consolidation).

These models extend the measurement-plane ContractSpec (agentassert_abc.models)
with the *enforcement* plane ported from agentassert-typec: real-time process
invariants, content operators, and the ALLOW/DENY/REDACT/MODIFY decision types.

Design (port-delta §D): the shared base models (ConstraintCheck, Invariants,
RecoveryConfig, ContractSpec, SatisfactionParams, DriftWeights, …) are REUSED
from agentassert_abc.models — never duplicated. This module only adds the new,
additive enforcement types and thin `*Extended` subclasses. All abc v2 field
constraints (delta ge=0.0, k ge=0, sum-1 weight validators, exactly-one-operator)
are inherited unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from agentassert_abc.models import (
    ContractSpec,
    Invariants,
    RecoveryConfig,
    _FrozenModel,
)

# --- Process Contract Operators (7 total, DSL v0.4) ---


class MustPrecede(_FrozenModel):
    """Ordering invariant: `before` tool/action must occur before `after`."""

    before: str
    after: str
    scope: Literal["turn", "session"] = "turn"


class MustState(_FrozenModel):
    """A field must be set before a matching tool call is permitted."""

    field: str
    before_tool_pattern: str
    rationale: str = ""


class ToolBlocklist(_FrozenModel):
    """Tools that must never be called within the given scope."""

    tools: list[str]
    scope: Literal["session", "turn"] = "session"


class ToolAllowlist(_FrozenModel):
    """Only these tools may be called within the given scope."""

    tools: list[str]
    scope: str = "session"


class ContextBudget(_FrozenModel):
    """Per-turn context-token budget with a breach action."""

    max_tokens_per_turn: int = Field(60_000, gt=0)
    action_on_breach: Literal["warn", "deny", "compress"] = "warn"


class ProcessDrift(_FrozenModel):
    """Tool-sequence distributional drift guard (JSD over a rolling window)."""

    window_size: int = Field(10, gt=0)
    jsd_threshold: float = Field(0.3, gt=0.0, le=1.0)
    action: Literal["log", "warn", "theta_penalty"] = "log"


class JudgePredicate(_FrozenModel):
    """LLM-as-judge predicate, sampled and cost-capped."""

    rubric: str
    sample_rate: float = Field(0.2, gt=0.0, le=1.0)
    model: str = "haiku"
    action_on_fail: Literal["log", "warn", "theta_penalty", "deny"] = "theta_penalty"
    cost_ceiling_usd_per_session: float = Field(0.10, ge=0.0)


# --- Content Operators (Phase 3) ---


class PiiPatternGroup(StrEnum):
    """Built-in PII pattern groups."""

    email = "email"
    phone = "phone"
    ssn = "ssn"
    credit_card = "credit_card"
    api_key = "api_key"
    ip_address = "ip_address"


class CustomPiiPattern(_FrozenModel):
    """User-supplied named PII regex."""

    name: str
    regex: str


class PiiFilter(_FrozenModel):
    """PII detection/redaction over agent output."""

    patterns: list[PiiPatternGroup] = [PiiPatternGroup.email, PiiPatternGroup.phone]
    action: Literal["log", "warn", "redact", "block"] = "log"
    streaming_action: Literal["log", "warn"] = "log"
    custom_patterns: list[CustomPiiPattern] = []


class ProviderPriceEntry(_FrozenModel):
    """USD-per-million-token prices for a provider."""

    input: float = Field(gt=0.0)
    output: float = Field(gt=0.0)


class CostCeiling(_FrozenModel):
    """Per-session spend ceiling with a breach action."""

    max_usd_per_session: float = Field(gt=0.0)
    action_on_breach: Literal["deny", "warn", "log"] = "warn"
    price_per_million_input: float | None = None
    price_per_million_output: float | None = None
    provider_price_map: dict[str, ProviderPriceEntry] = {}


class RepetitionGuard(_FrozenModel):
    """Detect and act on repeated identical tool-call sequences."""

    window_size: int = Field(5, ge=2, le=50)
    max_repeats: int = Field(3, ge=2, le=100)
    action: Literal["deny", "warn", "log"] = "deny"
    ignore_tools: list[str] = []


# --- Process Invariants container ---


class ProcessInvariants(_FrozenModel):
    """Container for all process-level + content operators (all optional)."""

    must_precede: list[MustPrecede] = []
    must_state: list[MustState] = []
    tool_blocklist: list[ToolBlocklist] = []
    tool_allowlist: list[ToolAllowlist] = []
    context_budget: ContextBudget | None = None
    process_drift: ProcessDrift | None = None
    judge_predicate: list[JudgePredicate] = []
    # Phase 3: content operators (None = disabled)
    pii_filter: PiiFilter | None = None
    cost_ceiling: CostCeiling | None = None
    repetition_guard: RepetitionGuard | None = None


def _list_to_process_invariants(ops: list[Any]) -> dict[str, Any]:
    """Coerce the legacy list-of-operators YAML form into the dict form.

    Older typec contracts express `process:` as a list of single-key dicts.
    This normalizes that shape so `ProcessInvariants` can validate it.
    """
    result: dict[str, Any] = {
        "must_precede": [],
        "must_state": [],
        "tool_blocklist": [],
        "tool_allowlist": [],
        "judge_predicate": [],
    }
    list_keys = {
        "must_precede",
        "must_state",
        "tool_blocklist",
        "tool_allowlist",
        "judge_predicate",
    }
    singular_keys = {
        "context_budget",
        "process_drift",
        "pii_filter",
        "cost_ceiling",
        "repetition_guard",
    }
    for op in ops:
        if not isinstance(op, dict):
            continue
        for key in list_keys:
            if key in op:
                result[key].append(op[key])
        for key in singular_keys:
            if key in op:
                result[key] = op[key]  # last one wins
    return result


# --- Extended containers (thin subclasses of abc v2 bases) ---


class InvariantsExtended(Invariants):
    """abc v2 Invariants (hard + soft) plus an optional process plane."""

    process: ProcessInvariants | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_process_from_list(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        process_raw = data.get("process")
        if isinstance(process_raw, list):
            # Immutability: copy rather than mutate the caller's dict.
            data = {**data, "process": _list_to_process_invariants(process_raw)}
        return data


class RecoveryConfigExtended(RecoveryConfig):
    """abc v2 RecoveryConfig plus typec's per-severity default actions."""

    on_hard_violation: str = "raise"
    on_soft_violation: str = "log_and_continue"


# --- Upstream provider URL overrides (proxy/gateway) ---


class UpstreamConfig(_FrozenModel):
    """Override the default LLM provider URLs the proxy forwards to.

    Priority: contract upstream > TYPEC_UPSTREAM_* env var
    > ANTHROPIC_BASE_URL/OPENAI_BASE_URL > built-in default.
    """

    anthropic: str | None = None
    openai: str | None = None
    gemini: str | None = None
    openrouter: str | None = None


# --- Root ContractSpecExtended ---


class ContractSpecExtended(ContractSpec):
    """ContractSpec + enforcement-plane extensions.

    Adds `dsl_version`, `upstream`, narrows `invariants` to `InvariantsExtended`
    (so `process:` parses) and `recovery` to `RecoveryConfigExtended`. Every
    other field and all validators are inherited unchanged from abc v2's
    `ContractSpec`.
    """

    dsl_version: str = "0.3"
    invariants: InvariantsExtended | None = None
    recovery: RecoveryConfigExtended | None = None
    upstream: UpstreamConfig | None = None


# --- Real-time enforcement decision types ---


class TypeCDecision(Enum):
    """Real-time enforcement decision for a single event."""

    ALLOW = "allow"
    MODIFY = "modify"
    DENY = "deny"
    REDACT = "redact"  # response allowed but must be redacted before return
    WARN = "warn"  # soft signal, not blocking


@dataclass(frozen=True)
class DecisionResult:
    """Typed envelope returned by the enforcer for each dispatched event."""

    decision: TypeCDecision
    reason: str = ""
    modified_args: dict[str, Any] | None = None
    violation_name: str = ""
    theta_penalty: float = 0.0

    def is_deny(self) -> bool:
        return self.decision == TypeCDecision.DENY

    def is_modify(self) -> bool:
        return self.decision == TypeCDecision.MODIFY

    def is_redact(self) -> bool:
        return self.decision == TypeCDecision.REDACT

    def is_warn(self) -> bool:
        return self.decision == TypeCDecision.WARN
