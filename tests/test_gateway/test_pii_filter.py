# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_pii_filter.py` — 14 tests minimum."""

from __future__ import annotations

from agentassert_abc.gateway.compiler import CompiledContract
from agentassert_abc.gateway.content.pii import apply_pii_redaction, evaluate_pii_filter
from agentassert_abc.gateway.content.pii_patterns import _PII_PATTERNS
from agentassert_abc.gateway.violation_log import ViolationLog
from agentassert_abc.process.models import (
    ContractSpecExtended,
    CustomPiiPattern,
    InvariantsExtended,
    PiiFilter,
    PiiPatternGroup,
    ProcessInvariants,
    TypeCDecision,
)


def _make_compiled(
    patterns: list[PiiPatternGroup],
    action: str = "log",
    streaming_action: str = "log",
    custom_patterns: list[CustomPiiPattern] | None = None,
) -> CompiledContract:
    pii_filter = PiiFilter(
        patterns=patterns,
        action=action,
        streaming_action=streaming_action,
        custom_patterns=custom_patterns or [],
    )
    spec = ContractSpecExtended(
        dsl_version="0.4",
        contractspec="typec/v0.4",
        kind="agent",
        name="test",
        description="test",
        version="0.1.0",
        invariants=InvariantsExtended(process=ProcessInvariants(pii_filter=pii_filter)),
    )
    return CompiledContract.from_spec(spec)


def test_no_pii_passes() -> None:
    compiled = _make_compiled([PiiPatternGroup.email])
    violations = ViolationLog()
    result = evaluate_pii_filter("Hello world, no PII here!", compiled, violations, False)
    assert result is None
    assert len(violations.all_violations()) == 0


def test_email_detected() -> None:
    compiled = _make_compiled([PiiPatternGroup.email], action="block")
    violations = ViolationLog()
    result = evaluate_pii_filter("Contact: user@example.com", compiled, violations, False)
    assert result is not None
    assert result.is_deny()
    assert "pii_filter" in result.violation_name


def test_phone_detected() -> None:
    compiled = _make_compiled([PiiPatternGroup.phone], action="block")
    violations = ViolationLog()
    result = evaluate_pii_filter("+1 (555) 123-4567", compiled, violations, False)
    assert result is not None
    assert result.is_deny()


def test_ssn_detected() -> None:
    compiled = _make_compiled([PiiPatternGroup.ssn], action="block")
    violations = ViolationLog()
    result = evaluate_pii_filter("SSN: 123-45-6789", compiled, violations, False)
    assert result is not None
    assert result.is_deny()


def test_credit_card_detected() -> None:
    compiled = _make_compiled([PiiPatternGroup.credit_card], action="block")
    violations = ViolationLog()
    result = evaluate_pii_filter("Card: 4111111111111111", compiled, violations, False)
    assert result is not None
    assert result.is_deny()


def test_api_key_detected() -> None:
    compiled = _make_compiled([PiiPatternGroup.api_key], action="block")
    violations = ViolationLog()
    result = evaluate_pii_filter(
        "My key: sk-AbCdEfGhIjKlMnOpQrStUvWx", compiled, violations, False
    )
    assert result is not None
    assert result.is_deny()


def test_action_log_does_not_deny() -> None:
    compiled = _make_compiled([PiiPatternGroup.email], action="log")
    violations = ViolationLog()
    result = evaluate_pii_filter("user@example.com", compiled, violations, False)
    assert result is None
    v = violations.all_violations()
    assert len(v) == 1
    assert v[0]["kind"] == "soft"


def test_action_block_returns_deny() -> None:
    compiled = _make_compiled([PiiPatternGroup.email], action="block")
    violations = ViolationLog()
    result = evaluate_pii_filter("user@example.com", compiled, violations, False)
    assert result is not None
    assert result.decision == TypeCDecision.DENY


def test_action_redact_returns_redact() -> None:
    compiled = _make_compiled([PiiPatternGroup.email], action="redact")
    violations = ViolationLog()
    result = evaluate_pii_filter("user@example.com", compiled, violations, False)
    assert result is not None
    assert result.decision == TypeCDecision.REDACT


def test_redact_replaces_pii() -> None:
    patterns = [("email", _PII_PATTERNS["email"])]
    text = "Contact: user@example.com and admin@corp.io"
    redacted = apply_pii_redaction(text, patterns)
    assert "user@example.com" not in redacted
    assert "admin@corp.io" not in redacted
    assert "[REDACTED:EMAIL]" in redacted


def test_streaming_block_degrades_to_warn() -> None:
    compiled = _make_compiled([PiiPatternGroup.email], action="block", streaming_action="warn")
    violations = ViolationLog()
    result = evaluate_pii_filter("user@example.com", compiled, violations, is_streaming=True)
    assert result is None
    v = violations.all_violations()
    assert len(v) == 1
    assert v[0]["kind"] == "soft"


def test_custom_pattern() -> None:
    custom = [CustomPiiPattern(name="project_id", regex=r"PROJ-[0-9]{6}")]
    compiled = _make_compiled([], action="block", custom_patterns=custom)
    violations = ViolationLog()
    result = evaluate_pii_filter("Working on PROJ-123456", compiled, violations, False)
    assert result is not None
    assert result.is_deny()


def test_empty_text_passes() -> None:
    compiled = _make_compiled([PiiPatternGroup.email], action="block")
    violations = ViolationLog()
    result = evaluate_pii_filter("", compiled, violations, False)
    assert result is None


def test_multiple_pii_types_detected() -> None:
    compiled = _make_compiled([PiiPatternGroup.email, PiiPatternGroup.phone], action="log")
    violations = ViolationLog()
    result = evaluate_pii_filter(
        "Email: user@example.com Phone: 555-123-4567", compiled, violations, False
    )
    assert result is None
    v = violations.all_violations()
    assert len(v) == 1
    assert "email" in v[0]["reason"]
    assert "phone" in v[0]["reason"]
