# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec
`packages/claude-code/tests/test_contract_templates.py`.

Uses `load_contract_extended` (not the base `parse_contract`) — these
templates are DSL v0.4 with `invariants.process` operators, which the base
`ContractSpec` parser silently drops (port-delta item #17). The extended
parser is the semantically correct one for enforcement-plane templates.
"""

from __future__ import annotations

from pathlib import Path

from agentassert_abc.dsl.parser import load_contract_extended

TEMPLATES = (
    Path(__file__).parent.parent.parent
    / "src"
    / "agentassert_abc"
    / "claude_code"
    / "contracts"
    / "templates"
)


def test_safety_minimal_valid() -> None:
    contract = load_contract_extended(TEMPLATES / "safety-minimal.yaml")
    assert contract.name == "safety-minimal"
    assert contract.invariants is not None
    assert contract.invariants.process is not None
    assert len(contract.invariants.process.tool_blocklist) == 1


def test_partner_mode_valid() -> None:
    contract = load_contract_extended(TEMPLATES / "partner-mode.yaml")
    assert contract.name == "partner-mode"
    assert contract.invariants is not None
    proc = contract.invariants.process
    assert proc is not None
    assert len(proc.must_precede) == 1
    assert len(proc.must_state) == 1
    assert proc.judge_predicate[0].model == "free-tier-model"


def test_full_governance_valid() -> None:
    contract = load_contract_extended(TEMPLATES / "full-governance.yaml")
    assert contract.name == "full-governance"
    proc = contract.invariants.process
    assert proc is not None
    assert len(proc.tool_allowlist) == 1
    assert proc.judge_predicate[0].model == "free-tier-model"


def test_templates_contain_no_leaked_internal_identifiers() -> None:
    """No internal infra naming (tap_*, ds-flash-free, mcp__hermes*) may
    ship in a published contract template.
    """
    banned = ("tap_", "ds-flash-free", "mcp__hermes")
    for template in TEMPLATES.glob("*.yaml"):
        text = template.read_text()
        for token in banned:
            assert token not in text, f"leaked identifier {token!r} in {template.name}"
