# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""ContractSpec DSL Parser — Layer 1.

Loads YAML contract files and produces validated, typed ContractSpec objects.
Two-stage validation: structural (Pydantic) + semantic (cross-reference checks).

Patent reference: TECHNICAL-ATTACHMENT.md §3.1 (Layer 1), §4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from agentassert_abc.dsl.models import ParseResult, ValidationError
from agentassert_abc.dsl.validator import validate_contract
from agentassert_abc.exceptions import ContractParseError, ContractValidationError
from agentassert_abc.models import ContractSpec

try:
    from ruamel.yaml import YAML, YAMLError
except ImportError as e:
    raise ImportError(
        "ruamel.yaml is required for YAML parsing. "
        "Install with: pip install agentassert-abc[yaml]"
    ) from e


def _parse_yaml(yaml_string: str) -> dict[str, Any]:
    """Parse YAML string to dict. Raises ContractParseError on syntax errors."""
    yaml = YAML(typ="safe", pure=True)
    try:
        data = yaml.load(yaml_string)
    except YAMLError as e:
        raise ContractParseError(f"YAML parse error: {e}") from e

    if not isinstance(data, dict):
        raise ContractParseError(
            f"Contract must be a YAML mapping, got {type(data).__name__}"
        )
    return data


def _validate_struct(data: dict[str, Any]) -> ContractSpec:
    """Structural validation via Pydantic. Raises ContractParseError."""
    try:
        return ContractSpec.model_validate(data)
    except PydanticValidationError as e:
        raise ContractParseError(f"Contract validation error: {e}") from e


def loads_contract(yaml_string: str) -> ContractSpec:
    """Load + validate contract from YAML string.

    Raises:
        ContractParseError: YAML syntax or structural validation failure.
        ContractValidationError: Semantic validation failure (errors only).
    """
    data = _parse_yaml(yaml_string)
    contract = _validate_struct(data)

    errors = validate_contract(contract)
    error_level = [e for e in errors if e.level == "error"]
    if error_level:
        msg = "; ".join(f"[{e.code}] {e.path}: {e.message}" for e in error_level)
        raise ContractValidationError(f"Semantic validation failed: {msg}")

    return contract


def load_contract(path: str | Path) -> ContractSpec:
    """Load + validate contract from YAML file.

    Raises:
        FileNotFoundError: Path does not exist.
        ContractParseError: YAML syntax or structural validation failure.
        ContractValidationError: Semantic validation failure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Contract file not found: {p}")
    return loads_contract(p.read_text(encoding="utf-8"))


def parses_contract(yaml_string: str) -> ParseResult:
    """Non-raising parse — returns ParseResult with errors."""
    try:
        data = _parse_yaml(yaml_string)
    except ContractParseError as e:
        return ParseResult(
            errors=[ValidationError(level="error", path="", message=str(e), code="YAML_ERROR")]
        )

    try:
        contract = _validate_struct(data)
    except ContractParseError as e:
        return ParseResult(
            errors=[ValidationError(level="error", path="", message=str(e), code="STRUCT_ERROR")]
        )

    sem_errors = validate_contract(contract)
    error_level = [e for e in sem_errors if e.level == "error"]
    if error_level:
        return ParseResult(errors=error_level)

    return ParseResult(contract=contract, errors=sem_errors)


def parse_contract(path: str | Path) -> ParseResult:
    """Non-raising parse from file."""
    p = Path(path)
    if not p.exists():
        return ParseResult(
            errors=[
                ValidationError(
                    level="error", path=str(p), message="File not found", code="FILE_NOT_FOUND"
                )
            ]
        )
    return parses_contract(p.read_text(encoding="utf-8"))
