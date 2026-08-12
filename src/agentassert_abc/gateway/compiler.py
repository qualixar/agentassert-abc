# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""CompiledContract — AST-compiles a ContractSpecExtended's process/content
operators into pre-compiled regex patterns and lookup structures.

Ported from agentassert-typec's `dsl/ast_compiler.py` (,
item #25). Import paths updated to reuse the already-ported
`agentassert_abc.process.models` (Phase A) instead of duplicating them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentassert_abc.gateway.content.pii_patterns import _PII_PATTERNS

if TYPE_CHECKING:
    from agentassert_abc.process.models import ContractSpecExtended


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a `*`/`?`-glob string into a case-insensitive regex."""
    escaped = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(escaped, re.IGNORECASE)


@dataclass
class CompiledContract:
    """Pre-compiled view of a ContractSpecExtended's enforcement plane."""

    spec: ContractSpecExtended

    tool_blocklist_patterns: list[re.Pattern[str]] = field(default_factory=list)
    tool_allowlist_patterns: list[tuple[str, list[re.Pattern[str]]]] = field(
        default_factory=list
    )
    must_precede_rules: list[dict[str, Any]] = field(default_factory=list)
    must_state_rules: list[dict[str, Any]] = field(default_factory=list)
    context_budget_limit: int | None = None
    context_budget_action: str = "warn"
    process_drift_config: Any | None = None  # ProcessDrift | None
    judge_predicates: list[dict[str, Any]] = field(default_factory=list)

    # Structural classification of the abc-plane hard/soft constraints —
    # informational only. Actual evaluation is done by
    # `agentassert_abc.evaluator.engine.evaluate()` against the live state
    # (see gateway/engine.py::_eval_post_action), which already dispatches
    # `expr` vs. structural checks internally via `evaluate_check()`.
    hard_checks: list[Any] = field(default_factory=list)
    soft_checks: list[Any] = field(default_factory=list)

    # Phase 3: content operators
    pii_compiled_patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list)
    pii_filter_config: Any | None = None  # PiiFilter | None
    cost_ceiling_config: Any | None = None  # CostCeiling | None
    repetition_guard_config: Any | None = None  # RepetitionGuard | None
    repetition_guard_ignore_patterns: list[re.Pattern[str]] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: ContractSpecExtended) -> CompiledContract:
        compiled = cls(spec=spec)
        compiled._compile_process_invariants()
        compiled._compile_abc_checks()
        return compiled

    def _compile_process_invariants(self) -> None:
        if not self.spec.invariants or not self.spec.invariants.process:
            return
        proc = self.spec.invariants.process

        for blocklist in proc.tool_blocklist:
            for pattern_str in blocklist.tools:
                for part in pattern_str.split("|"):
                    self.tool_blocklist_patterns.append(_glob_to_regex(part))

        for allowlist in proc.tool_allowlist:
            compiled_patterns = [_glob_to_regex(t) for t in allowlist.tools]
            self.tool_allowlist_patterns.append((allowlist.scope, compiled_patterns))

        if proc.context_budget:
            self.context_budget_limit = proc.context_budget.max_tokens_per_turn
            self.context_budget_action = proc.context_budget.action_on_breach

        for mp in proc.must_precede:
            self.must_precede_rules.append(
                {"before": mp.before, "after": mp.after, "scope": mp.scope}
            )

        for ms in proc.must_state:
            patterns = [_glob_to_regex(p) for p in ms.before_tool_pattern.split("|")]
            self.must_state_rules.append(
                {"field": ms.field, "patterns": patterns, "rationale": ms.rationale}
            )

        if proc.process_drift:
            self.process_drift_config = proc.process_drift

        for jp in proc.judge_predicate:
            self.judge_predicates.append(
                {
                    "rubric": jp.rubric,
                    "sample_rate": jp.sample_rate,
                    "model": jp.model,
                    "action_on_fail": jp.action_on_fail,
                    "cost_ceiling": jp.cost_ceiling_usd_per_session,
                }
            )

        self._compile_pii_filter(proc)

        if proc.cost_ceiling:
            self.cost_ceiling_config = proc.cost_ceiling

        if proc.repetition_guard:
            self.repetition_guard_config = proc.repetition_guard
            for tool in proc.repetition_guard.ignore_tools:
                self.repetition_guard_ignore_patterns.append(_glob_to_regex(tool))

    def _compile_pii_filter(self, proc: Any) -> None:
        if not proc.pii_filter:
            return
        self.pii_filter_config = proc.pii_filter
        for group in proc.pii_filter.patterns:
            rx = re.compile(_PII_PATTERNS[group.value].pattern, re.IGNORECASE)
            self.pii_compiled_patterns.append((group.value, rx))
        for custom in proc.pii_filter.custom_patterns:
            self.pii_compiled_patterns.append((custom.name, re.compile(custom.regex)))

    def _compile_abc_checks(self) -> None:
        """Classify invariants.hard/soft as ('expr'|'struct', name, payload).

        Informational/test-parity only (see `hard_checks`/`soft_checks`
        docstring above) — the enforcer evaluates the real ConstraintCheck
        objects via `agentassert_abc.evaluator.engine.evaluate()`.
        """
        if not self.spec.invariants:
            return
        for c in self.spec.invariants.hard:
            if c.check.expr:
                self.hard_checks.append(("expr", c.name, c.check.expr))
            else:
                self.hard_checks.append(("struct", c.name, c.check))
        for c in self.spec.invariants.soft:
            if c.check.expr:
                self.soft_checks.append(("expr", c.name, c.check.expr))
            else:
                self.soft_checks.append(("struct", c.name, c.check))
