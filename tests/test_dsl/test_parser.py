# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for ContractSpec DSL parser — Layer 1.

Patent reference: TECHNICAL-ATTACHMENT.md §3.1, §4.
Tests cover: YAML loading, structural validation, alias handling, error cases.
"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class TestLoadFromString:
    """Load contracts from YAML strings."""

    def test_minimal_contract(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract

        yaml = """
contractspec: "0.1"
kind: agent
name: test
description: A test contract
version: "1.0.0"
"""
        contract = loads_contract(yaml)
        assert contract.kind == "agent"
        assert contract.name == "test"
        assert contract.contractspec == "0.1"

    def test_contract_with_preconditions(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract

        yaml = """
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
preconditions:
  - name: session-valid
    check:
      field: session.active
      equals: true
"""
        contract = loads_contract(yaml)
        assert len(contract.preconditions) == 1
        assert contract.preconditions[0].name == "session-valid"
        assert contract.preconditions[0].check.equals is True

    def test_contract_with_hard_constraints(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract

        yaml = """
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: no-pii
      category: confidentiality
      check:
        field: output.pii_detected
        equals: false
"""
        contract = loads_contract(yaml)
        assert contract.invariants is not None
        assert len(contract.invariants.hard) == 1
        assert contract.invariants.hard[0].name == "no-pii"

    def test_contract_with_soft_constraints_and_recovery(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract

        yaml = """
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  soft:
    - name: brand-tone
      check:
        field: output.tone_score
        gte: 0.7
      recovery: fix-tone
      recovery_window: 2
recovery:
  strategies:
    - name: fix-tone
      type: inject_correction
      actions:
        - "Fix the tone"
      max_attempts: 2
"""
        contract = loads_contract(yaml)
        assert contract.invariants is not None
        assert len(contract.invariants.soft) == 1
        assert contract.invariants.soft[0].recovery == "fix-tone"
        assert contract.recovery is not None
        assert len(contract.recovery.strategies) == 1

    def test_satisfaction_params(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract

        yaml = """
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
satisfaction:
  p: 0.99
  delta: 0.05
  k: 5
"""
        contract = loads_contract(yaml)
        assert contract.satisfaction is not None
        assert contract.satisfaction.p == 0.99
        assert contract.satisfaction.delta == 0.05
        assert contract.satisfaction.k == 5

    def test_drift_config(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract

        yaml = """
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
drift:
  weights:
    compliance: 0.7
    distributional: 0.3
  window: 100
  thresholds:
    warning: 0.2
    critical: 0.5
"""
        contract = loads_contract(yaml)
        assert contract.drift is not None
        assert contract.drift.weights.compliance == 0.7
        assert contract.drift.window == 100

    def test_in_operator_alias(self) -> None:
        """The 'in' YAML key must map to 'in_' Python field."""
        from agentassert_abc.dsl.parser import loads_contract

        yaml = """
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
invariants:
  hard:
    - name: category-check
      check:
        field: output.category
        in:
          - electronics
          - clothing
"""
        contract = loads_contract(yaml)
        assert contract.invariants is not None
        check = contract.invariants.hard[0].check
        assert check.in_ == ["electronics", "clothing"]

    def test_pipeline_kind(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract

        yaml = """
contractspec: "0.1"
kind: pipeline
name: test-pipeline
description: A pipeline contract
version: "1.0.0"
"""
        contract = loads_contract(yaml)
        assert contract.kind == "pipeline"


class TestLoadFromFile:
    """Load contracts from YAML files."""

    def test_load_from_path(self, tmp_path: Path) -> None:
        from agentassert_abc.dsl.parser import load_contract

        f = tmp_path / "contract.yaml"
        f.write_text("""
contractspec: "0.1"
kind: agent
name: file-test
description: from file
version: "1.0.0"
""")
        contract = load_contract(f)
        assert contract.name == "file-test"

    def test_load_from_string_path(self, tmp_path: Path) -> None:
        from agentassert_abc.dsl.parser import load_contract

        f = tmp_path / "contract.yaml"
        f.write_text("""
contractspec: "0.1"
kind: agent
name: string-path
description: from string path
version: "1.0.0"
""")
        contract = load_contract(str(f))
        assert contract.name == "string-path"

    def test_file_not_found(self) -> None:
        from agentassert_abc.dsl.parser import load_contract

        with pytest.raises(FileNotFoundError):
            load_contract("/nonexistent/contract.yaml")


class TestEcommerceContract:
    """Integration test: parse the EXACT e-commerce contract from patent §4.1."""

    def test_full_ecommerce_contract(self) -> None:
        from agentassert_abc.dsl.parser import load_contract

        contract = load_contract(FIXTURES / "ecommerce-product-recommendation.yaml")

        # Header
        assert contract.contractspec == "0.1"
        assert contract.kind == "agent"
        assert contract.name == "ecommerce-product-recommendation"

        # Metadata
        assert contract.metadata is not None
        assert contract.metadata.domain == "ecommerce"

        # Preconditions (3)
        assert len(contract.preconditions) == 3

        # Hard constraints (6)
        assert contract.invariants is not None
        assert len(contract.invariants.hard) == 6
        hard_names = [c.name for c in contract.invariants.hard]
        assert "no-competitor-products" in hard_names
        assert "no-pii-leak" in hard_names

        # Soft constraints (6)
        assert len(contract.invariants.soft) == 6
        soft_names = [c.name for c in contract.invariants.soft]
        assert "brand-tone" in soft_names
        assert "response-latency" in soft_names

        # Governance hard (1) + soft (2)
        assert contract.governance is not None
        assert len(contract.governance.hard) == 1
        assert len(contract.governance.soft) == 2

        # Recovery strategies (8)
        assert contract.recovery is not None
        assert len(contract.recovery.strategies) == 8

        # Satisfaction (patent defaults)
        assert contract.satisfaction is not None
        assert contract.satisfaction.p == 0.95
        assert contract.satisfaction.delta == 0.1
        assert contract.satisfaction.k == 3

        # Drift (patent defaults)
        assert contract.drift is not None
        assert contract.drift.weights.compliance == 0.6
        assert contract.drift.weights.distributional == 0.4
        assert contract.drift.window == 50

        # Reliability (patent defaults)
        assert contract.reliability is not None
        assert contract.reliability.weights.compliance == 0.35
        assert contract.reliability.deployment_threshold == 0.90


class TestParseErrors:
    """Error handling for invalid inputs."""

    def test_invalid_yaml_syntax(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.exceptions import ContractParseError

        with pytest.raises(ContractParseError):
            loads_contract("{{invalid yaml::")

    def test_missing_required_fields(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.exceptions import ContractParseError

        with pytest.raises(ContractParseError):
            loads_contract("contractspec: '0.1'\nkind: agent\n")

    def test_invalid_kind(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.exceptions import ContractParseError

        yaml = """
contractspec: "0.1"
kind: invalid
name: test
description: test
version: "1.0.0"
"""
        with pytest.raises(ContractParseError):
            loads_contract(yaml)

    def test_invalid_recovery_type(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.exceptions import ContractParseError

        yaml = """
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
recovery:
  strategies:
    - name: bad
      type: nonexistent_type
"""
        with pytest.raises(ContractParseError):
            loads_contract(yaml)

    def test_non_dict_yaml(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.exceptions import ContractParseError

        with pytest.raises(ContractParseError):
            loads_contract("- just\n- a\n- list\n")


class TestParseResult:
    """Non-raising parse API."""

    def test_parse_valid_returns_contract(self) -> None:
        from agentassert_abc.dsl.parser import parses_contract

        yaml = """
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
"""
        result = parses_contract(yaml)
        assert result.is_valid
        assert result.contract is not None
        assert result.contract.name == "test"

    def test_parse_invalid_returns_errors(self) -> None:
        from agentassert_abc.dsl.parser import parses_contract

        result = parses_contract("{{bad yaml")
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.contract is None
