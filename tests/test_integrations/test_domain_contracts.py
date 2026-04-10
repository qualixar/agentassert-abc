# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for domain contract YAML files — all 12 must parse and validate.

These tests verify that every contract in contracts/examples/ is:
1. Valid YAML parseable by ruamel.yaml
2. Valid ContractSpec (Pydantic model)
3. Passes semantic validation with zero errors
4. Has the expected constraint counts from patent §10
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agentassert_abc as aa
from agentassert_abc.integrations.generic import GenericAdapter

CONTRACTS_DIR = Path(__file__).parent.parent.parent / "contracts" / "examples"


def _contract_files() -> list[Path]:
    """List all contract YAML files."""
    return sorted(CONTRACTS_DIR.glob("*.yaml"))


class TestAllContractsParse:
    """Every contract file must parse into a valid ContractSpec."""

    @pytest.mark.parametrize("yaml_file", _contract_files(), ids=lambda p: p.stem)
    def test_contract_parses(self, yaml_file: Path) -> None:
        contract = aa.load(str(yaml_file))
        assert contract is not None
        assert contract.name != ""
        assert contract.contractspec == "0.1"

    @pytest.mark.parametrize("yaml_file", _contract_files(), ids=lambda p: p.stem)
    def test_contract_validates_clean(self, yaml_file: Path) -> None:
        """Zero semantic validation errors."""
        contract = aa.load(str(yaml_file))
        errors = aa.validate(contract)
        error_list = [e for e in errors if e.level == "error"]
        assert len(error_list) == 0, f"Validation errors: {[e.message for e in error_list]}"


class TestContractConstraintCounts:
    """Verify constraint counts match patent §10 (minimum)."""

    def _load(self, name: str) -> aa.ContractSpec:
        return aa.load(str(CONTRACTS_DIR / f"{name}.yaml"))

    def _counts(self, contract: aa.ContractSpec) -> tuple[int, int]:
        hard = len(contract.invariants.hard) if contract.invariants else 0
        soft = len(contract.invariants.soft) if contract.invariants else 0
        if contract.governance:
            hard += len(contract.governance.hard)
            soft += len(contract.governance.soft)
        return hard, soft

    def test_ecommerce_product_recommendation(self) -> None:
        h, s = self._counts(self._load("ecommerce-product-recommendation"))
        assert h >= 7 and s >= 8  # Patent: 7 hard, 8 soft

    def test_ecommerce_order_management(self) -> None:
        h, s = self._counts(self._load("ecommerce-order-management"))
        assert h >= 6 and s >= 6  # Patent: 6 hard, 6 soft

    def test_ecommerce_customer_service(self) -> None:
        h, s = self._counts(self._load("ecommerce-customer-service"))
        assert h >= 6 and s >= 6  # Patent: 6 hard, 6 soft

    def test_financial_advisor(self) -> None:
        h, s = self._counts(self._load("financial-advisor"))
        assert h >= 6 and s >= 6  # Patent: 6 hard, 6 soft

    def test_healthcare_triage(self) -> None:
        h, s = self._counts(self._load("healthcare-triage"))
        assert h >= 8 and s >= 5  # Patent: 8 hard, 5 soft

    def test_retail_shopping_assistant(self) -> None:
        h, s = self._counts(self._load("retail-shopping-assistant"))
        assert h >= 6 and s >= 7  # Patent: 6 hard, 7 soft

    def test_telecom_customer_support(self) -> None:
        h, s = self._counts(self._load("telecom-customer-support"))
        assert h >= 6 and s >= 7  # Patent: 6 hard, 7 soft

    def test_code_generation(self) -> None:
        h, s = self._counts(self._load("code-generation"))
        assert h >= 6 and s >= 5  # Patent: 6 hard, 5 soft

    def test_research_assistant(self) -> None:
        h, s = self._counts(self._load("research-assistant"))
        assert h >= 5 and s >= 5  # Patent: 5 hard, 5 soft

    def test_customer_support(self) -> None:
        h, s = self._counts(self._load("customer-support"))
        assert h >= 5 and s >= 4  # Patent: 5 hard, 4 soft

    def test_mcp_tool_server(self) -> None:
        """2026-specific: MCP tool server contract."""
        h, s = self._counts(self._load("mcp-tool-server"))
        assert h >= 5 and s >= 4

    def test_rag_agent(self) -> None:
        """2026-specific: RAG agent contract."""
        h, s = self._counts(self._load("rag-agent"))
        assert h >= 6 and s >= 5


class TestContractsWorkWithAdapter:
    """Each contract can be used with GenericAdapter for monitoring."""

    @pytest.mark.parametrize("yaml_file", _contract_files(), ids=lambda p: p.stem)
    def test_adapter_accepts_contract(self, yaml_file: Path) -> None:
        contract = aa.load(str(yaml_file))
        adapter = GenericAdapter(contract)
        # Minimal state — will have violations, but should not crash
        result = adapter.check({"dummy_field": True})
        assert result is not None
        summary = adapter.session_summary()
        assert summary.turn_count == 1
