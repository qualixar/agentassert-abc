# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for A2A Compliance Bridge."""

from agentassert_abc.integrations.a2a import A2AComplianceBridge, A2AComplianceResult


class TestA2AComplianceBridge:
    """A2A agent card → ContractSpec bridge tests."""

    def test_from_agent_card_basic(self) -> None:
        bridge = A2AComplianceBridge()
        card = bridge.agent_card_template(
            name="search-agent",
            description="Web search agent",
            capabilities=["web_search"],
            safety_guarantees=["no_pii", "no_hallucination"],
            rate_limit_rpm=60,
        )
        result = bridge.from_agent_card(card)
        contract = result["contract"]
        compliance = result["compliance"]

        assert contract.name == "a2a-search-agent"
        assert len(contract.invariants.hard) == 2
        assert contract.governance is not None
        assert len(contract.governance.soft) == 1
        assert compliance.compliant is True

    def test_from_agent_card_no_safety(self) -> None:
        bridge = A2AComplianceBridge()
        card = bridge.agent_card_template(
            name="simple-agent", description="Minimal",
            capabilities=["chat"],
        )
        result = bridge.from_agent_card(card)
        compliance = result["compliance"]
        assert len(compliance.warnings) == 1

    def test_from_agent_card_with_input_schema(self) -> None:
        bridge = A2AComplianceBridge()
        card = bridge.agent_card_template(
            name="processor", description="Data processor",
            capabilities=["transform"],
        )
        card["inputSchema"] = {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        }
        result = bridge.from_agent_card(card)
        contract = result["contract"]
        preconditions = contract.preconditions
        assert len(preconditions) == 1
        assert preconditions[0].name == "input-filename"

    def test_verify_compliant(self) -> None:
        bridge = A2AComplianceBridge()
        card = bridge.agent_card_template(
            name="good-agent", description="", capabilities=["x"],
        )
        result = bridge.from_agent_card(card)
        contract = result["contract"]
        compliance = bridge.verify(contract, {
            "theta": 0.93, "c_bar": 0.96, "d_bar": 0.04,
            "violations": 0, "recoveries": 1, "recovery_success": 1,
            "turns": 10,
        })
        assert compliance.compliant is True

    def test_verify_noncompliant_low_theta(self) -> None:
        bridge = A2AComplianceBridge()
        card = bridge.agent_card_template(
            name="bad-agent", description="", capabilities=["x"],
        )
        result = bridge.from_agent_card(card)
        contract = result["contract"]
        compliance = bridge.verify(contract, {
            "theta": 0.82, "c_bar": 0.80, "d_bar": 0.20,
            "violations": 5, "recoveries": 5, "recovery_success": 1,
            "turns": 10,
        })
        assert compliance.compliant is False


class TestA2AComplianceResult:
    def test_defaults(self) -> None:
        r = A2AComplianceResult()
        assert r.compliant is True
        assert r.missing_capabilities == []
