# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""A2A (Agent-to-Agent) Compliance Bridge — Google A2A protocol ↔ ContractSpec.

Maps A2A agent cards to AgentAssert behavioral contracts and verifies
compliance at the protocol handshake level. Supports IACS (Inter-Agent
Compliance Specification) extensions for enterprise agent mesh deployments.

Phase 7 — Integration & Marketplace → A2A Compliance Bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentassert_abc.models import (
    ConstraintCheck,
    ContractSpec,
    Governance,
    GovernanceConstraint,
    HardConstraint,
    Invariants,
    Precondition,
    SoftConstraint,
)


@dataclass(frozen=True)
class A2AComplianceResult:
    """Result of A2A agent card compliance verification.

    Attributes:
        agent_card_id: The A2A agent card's unique identifier.
        compliant: True if all required contract constraints are satisfied.
        contract_name: Generated ContractSpec name.
        missing_capabilities: Capabilities declared in card but not covered
            by the generated contract.
        warnings: Non-blocking compliance concerns.
    """

    agent_card_id: str = ""
    compliant: bool = True
    contract_name: str = ""
    missing_capabilities: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class A2AComplianceBridge:
    """Bridge Google A2A agent cards to AgentAssert ContractSpec.

    Parses an A2A agent card (JSON per A2A spec) and generates a
    corresponding ContractSpec with:
    - Preconditions from required input schema fields
    - Hard constraints from declared safety guarantees
    - Soft constraints from capability quality assertions
    - Governance constraints from rate/budget limits

    Usage:
        bridge = A2AComplianceBridge()
        card = bridge.from_agent_card(a2a_card_json)
        contract = card["contract"]  # ContractSpec
        compliance = bridge.verify(contract, session_data)
    """

    # A2A card schema fields we recognize and map
    _SAFETY_FIELDS = {
        "no_hallucination": "output.hallucination_detected",
        "no_pii": "output.pii_detected",
        "no_toxicity": "output.toxicity_score",
        "factual_only": "output.factual_accuracy",
        "schema_conformance": "output.schema_valid",
    }

    def from_agent_card(self, card: dict[str, Any]) -> dict[str, Any]:
        """Parse A2A agent card into a ContractSpec + metadata.

        Args:
            card: A2A agent card JSON dict. Expected keys:
                name, description, capabilities, inputSchema (optional),
                safety (optional), rateLimit (optional).

        Returns:
            {"contract": ContractSpec, "compliance": A2AComplianceResult}
        """
        name = card.get("name", "unknown-agent")
        desc = card.get("description", "")
        capabilities = card.get("capabilities", {})
        safety = card.get("safety", {})
        rate_limit = card.get("rateLimit", {})
        input_schema = card.get("inputSchema", {})

        # Preconditions from input schema
        preconditions: list[Precondition] = []
        if input_schema:
            required = input_schema.get("required", [])
            for field_name in required:
                preconditions.append(
                    Precondition(
                        name=f"input-{field_name}",
                        description=f"Required input: {field_name}",
                        check=ConstraintCheck(
                            field=f"input.{field_name}", exists=True,
                        ),
                    )
                )

        # Hard constraints from declared safety guarantees
        hard: list[HardConstraint] = []
        for saf_key, saf_val in safety.items():
            field = self._SAFETY_FIELDS.get(saf_key, saf_key)
            if isinstance(saf_val, bool) and saf_val:
                hard.append(
                    HardConstraint(
                        name=f"safety-{saf_key}",
                        description=f"A2A safety guarantee: {saf_key}",
                        category="safety",
                        check=ConstraintCheck(field=field, equals=False),
                    )
                )

        # Soft constraints from capability quality
        soft: list[SoftConstraint] = []
        if capabilities:
            for cap_name, cap_spec in capabilities.items():
                if isinstance(cap_spec, dict) and "quality_threshold" in cap_spec:
                    soft.append(
                        SoftConstraint(
                            name=f"quality-{cap_name}",
                            description=f"Capability quality: {cap_name}",
                            category="quality",
                            check=ConstraintCheck(
                                field=f"output.{cap_name}_quality",
                                gte=cap_spec["quality_threshold"],
                            ),
                            recovery="refactor",
                        )
                    )

        # Governance from rate limit
        governance: list[GovernanceConstraint] = []
        if rate_limit:
            max_req = rate_limit.get("requestsPerMinute", 0)
            if max_req > 0:
                governance.append(
                    GovernanceConstraint(
                        name="rate-limit",
                        check=ConstraintCheck(
                            field="governance.request_rate", lte=max_req,
                        ),
                    )
                )

        gov = Governance(soft=governance) if governance else None
        contract = ContractSpec(
            contractspec="0.1",
            kind="agent",
            name=f"a2a-{name}",
            description=f"A2A bridged: {desc}",
            version="1.0.0",
            preconditions=preconditions,
            invariants=Invariants(hard=hard, soft=soft),
            governance=gov,
        )

        compliance = A2AComplianceResult(
            agent_card_id=card.get("id", name),
            compliant=True,
            contract_name=contract.name,
            missing_capabilities=[],
            warnings=(
                ["No safety guarantees declared in A2A card"]
                if not safety else []
            ),
        )

        return {"contract": contract, "compliance": compliance}

    def verify(
        self,
        contract: ContractSpec,
        session_data: dict[str, Any],
    ) -> A2AComplianceResult:
        """Verify A2A compliance from observed session data.

        Args:
            contract: The ContractSpec generated by from_agent_card().
            session_data: Dict with keys: theta, c_bar, d_bar, violations,
                recoveries, recovery_success, turns.

        Returns:
            A2AComplianceResult with compliance verdict.
        """
        theta = session_data.get("theta", 0.0)
        recovery_rate = (
            session_data.get("recovery_success", 0)
            / max(session_data.get("recoveries", 1), 1)
        )

        warnings: list[str] = []
        if theta < 0.90:
            warnings.append(
                f"Theta {theta:.3f} below deployment threshold 0.90"
            )
        if recovery_rate < 0.5:
            warnings.append(
                f"Recovery rate {recovery_rate:.2f} below 0.50 — "
                "human oversight recommended (A2A §4.3)"
            )

        return A2AComplianceResult(
            agent_card_id=contract.name,
            compliant=theta >= 0.90 and len(warnings) == 0,
            contract_name=contract.name,
            warnings=warnings,
        )

    @staticmethod
    def agent_card_template(
        name: str,
        description: str,
        capabilities: list[str],
        safety_guarantees: list[str] | None = None,
        rate_limit_rpm: int = 0,
    ) -> dict[str, Any]:
        """Generate a minimal valid A2A agent card dict for testing."""
        card: dict[str, Any] = {
            "name": name,
            "description": description,
            "capabilities": {
                c: {"quality_threshold": 0.7} for c in capabilities
            },
        }
        if safety_guarantees:
            card["safety"] = {s: True for s in safety_guarantees}
        if rate_limit_rpm > 0:
            card["rateLimit"] = {"requestsPerMinute": rate_limit_rpm}
        return card
