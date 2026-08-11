# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_ast_compiler.py` +
`tests/test_coverage_gaps.py::TestAstCompilerABCSoftChecks`.

`tap_*`/`mcp__hermes` fixtures genericized to `paid_api_*`/`mcp__paid_tool`
per the port IP policy (port-delta §E.3).
"""

from __future__ import annotations

import pytest

from agentassert_abc.gateway.compiler import CompiledContract
from agentassert_abc.models import ConstraintCheck, HardConstraint, SoftConstraint
from agentassert_abc.process.models import (
    ContextBudget,
    ContractSpecExtended,
    InvariantsExtended,
    JudgePredicate,
    MustPrecede,
    MustState,
    ProcessDrift,
    ProcessInvariants,
    ToolAllowlist,
    ToolBlocklist,
)


@pytest.fixture
def minimal_spec() -> ContractSpecExtended:
    return ContractSpecExtended(
        dsl_version="0.4",
        contractspec="1.0",
        kind="agent",
        name="test",
        description="test",
        version="0.1",
    )


class TestCompiledContract:
    def test_empty_spec(self, minimal_spec: ContractSpecExtended) -> None:
        cc = CompiledContract.from_spec(minimal_spec)
        assert cc.tool_blocklist_patterns == []
        assert cc.must_state_rules == []
        assert cc.must_precede_rules == []
        assert cc.context_budget_limit is None
        assert cc.judge_predicates == []

    def test_tool_blocklist_compile(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="blocklist-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    tool_blocklist=[ToolBlocklist(tools=["rm -rf /*", "curl|bash", "mkfs.*"])]
                )
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert len(cc.tool_blocklist_patterns) == 4  # 1 + 2 alternates + 1

    def test_blocklist_pattern_matches(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="blocklist-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(tool_blocklist=[ToolBlocklist(tools=["rm"])])
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert len(cc.tool_blocklist_patterns) == 1
        pattern = cc.tool_blocklist_patterns[0]
        assert pattern.match("rm") is not None
        assert pattern.match("Read") is None

    def test_blocklist_wildcard_match(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="wildcard-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(tool_blocklist=[ToolBlocklist(tools=["paid_api_*"])])
            ),
        )
        cc = CompiledContract.from_spec(spec)
        pattern = cc.tool_blocklist_patterns[0]
        assert pattern.search("paid_api_deepseek") is not None
        assert pattern.search("mcp__paid_tool") is None

    def test_blocklist_case_insensitive(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="case-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(tool_blocklist=[ToolBlocklist(tools=["Rm"])])
            ),
        )
        cc = CompiledContract.from_spec(spec)
        pattern = cc.tool_blocklist_patterns[0]
        assert pattern.match("rm") is not None
        assert pattern.match("RM") is not None
        assert pattern.match("Rm") is not None

    def test_allowlist_compile(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="allowlist-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    tool_allowlist=[ToolAllowlist(tools=["Read", "Write", "Edit"])]
                )
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert len(cc.tool_allowlist_patterns) == 1
        scope, patterns = cc.tool_allowlist_patterns[0]
        assert scope == "session"
        assert len(patterns) == 3

    def test_must_state_compile(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="must-state-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    must_state=[
                        MustState(
                            field="cost",
                            before_tool_pattern="paid_api_deepseek_*|paid_api_anthropic_*",
                            rationale="Cost required",
                        )
                    ]
                )
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert len(cc.must_state_rules) == 1
        rule = cc.must_state_rules[0]
        assert rule["field"] == "cost"
        assert len(rule["patterns"]) == 2
        assert rule["patterns"][0].search("paid_api_deepseek_v4") is not None
        assert rule["patterns"][1].search("paid_api_anthropic_haiku") is not None

    def test_context_budget_compile(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="budget-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    context_budget=ContextBudget(
                        max_tokens_per_turn=30000, action_on_breach="deny"
                    )
                )
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert cc.context_budget_limit == 30000
        assert cc.context_budget_action == "deny"

    def test_must_precede_compile(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="precede-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    must_precede=[
                        MustPrecede(before="challenge", after="recommendation", scope="turn")
                    ]
                )
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert len(cc.must_precede_rules) == 1
        assert cc.must_precede_rules[0]["before"] == "challenge"
        assert cc.must_precede_rules[0]["after"] == "recommendation"

    def test_process_drift_compile(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="drift-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    process_drift=ProcessDrift(window_size=20, jsd_threshold=0.4, action="warn")
                )
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert cc.process_drift_config is not None
        assert cc.process_drift_config.window_size == 20

    def test_judge_predicate_compile(self) -> None:
        spec = ContractSpecExtended(
            dsl_version="0.4",
            contractspec="1.0",
            kind="agent",
            name="judge-test",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                process=ProcessInvariants(
                    judge_predicate=[
                        JudgePredicate(
                            rubric="Concrete recommendations only",
                            sample_rate=0.25,
                            model="free-tier-model",
                            action_on_fail="theta_penalty",
                            cost_ceiling_usd_per_session=0.05,
                        )
                    ]
                )
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert len(cc.judge_predicates) == 1
        jp = cc.judge_predicates[0]
        assert jp["rubric"] == "Concrete recommendations only"
        assert jp["sample_rate"] == 0.25
        assert jp["model"] == "free-tier-model"


class TestCompiledContractAbcChecks:
    """hard_checks/soft_checks classification — informational/test-parity only."""

    def test_soft_expr_check_compiled(self) -> None:
        spec = ContractSpecExtended(
            contractspec="1.0",
            kind="agent",
            name="soft-expr",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                soft=[
                    SoftConstraint(
                        name="latency",
                        check=ConstraintCheck(field="latency_ms", expr="latency_ms < 5000"),
                    )
                ]
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert len(cc.soft_checks) == 1
        assert cc.soft_checks[0][0] == "expr"

    def test_soft_struct_check_compiled(self) -> None:
        spec = ContractSpecExtended(
            contractspec="1.0",
            kind="agent",
            name="soft-struct",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                soft=[
                    SoftConstraint(name="fast", check=ConstraintCheck(field="latency_ms", lt=5000))
                ]
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert len(cc.soft_checks) == 1
        assert cc.soft_checks[0][0] == "struct"

    def test_hard_expr_check_compiled(self) -> None:
        spec = ContractSpecExtended(
            contractspec="1.0",
            kind="agent",
            name="hard-expr",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                hard=[
                    HardConstraint(
                        name="no-pii", check=ConstraintCheck(field="pii", expr="pii == 0")
                    )
                ]
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert len(cc.hard_checks) == 1
        assert cc.hard_checks[0][0] == "expr"

    def test_hard_struct_check_compiled(self) -> None:
        spec = ContractSpecExtended(
            contractspec="1.0",
            kind="agent",
            name="hard-struct",
            description="test",
            version="0.1",
            invariants=InvariantsExtended(
                hard=[
                    HardConstraint(name="no-pii", check=ConstraintCheck(field="pii", equals=False))
                ]
            ),
        )
        cc = CompiledContract.from_spec(spec)
        assert len(cc.hard_checks) == 1
        assert cc.hard_checks[0][0] == "struct"
