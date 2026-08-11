# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_cost_ceiling.py` — 13 tests minimum.

`_extract_usage`/`_update_cost` were made public (`extract_usage`/`update_cost`)
during the port since they are legitimate reusable content-op helpers, not
private implementation details.
"""

from __future__ import annotations

from agentassert_abc.gateway.compiler import CompiledContract
from agentassert_abc.gateway.content.cost import (
    evaluate_cost_ceiling,
    extract_usage,
    update_cost,
)
from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.gateway.events import PreAction
from agentassert_abc.gateway.persistence import SessionStore
from agentassert_abc.gateway.violation_log import ViolationLog
from agentassert_abc.process.models import (
    ContractSpecExtended,
    CostCeiling,
    InvariantsExtended,
    ProcessInvariants,
    ProviderPriceEntry,
    TypeCDecision,
)


def _make_compiled(
    max_usd: float,
    action: str = "deny",
    provider_price_map: dict[str, tuple[float, float]] | None = None,
    price_per_million_input: float | None = None,
    price_per_million_output: float | None = None,
) -> CompiledContract:
    pm = {}
    if provider_price_map:
        for k, (inp, out) in provider_price_map.items():
            pm[k] = ProviderPriceEntry(input=inp, output=out)

    ceiling = CostCeiling(
        max_usd_per_session=max_usd,
        action_on_breach=action,
        price_per_million_input=price_per_million_input,
        price_per_million_output=price_per_million_output,
        provider_price_map=pm,
    )
    spec = ContractSpecExtended(
        dsl_version="0.4",
        contractspec="typec/v0.4",
        kind="agent",
        name="test",
        description="test",
        version="0.1.0",
        invariants=InvariantsExtended(process=ProcessInvariants(cost_ceiling=ceiling)),
    )
    return CompiledContract.from_spec(spec)


def _make_compiled_no_ceiling() -> CompiledContract:
    spec = ContractSpecExtended(
        dsl_version="0.4",
        contractspec="typec/v0.4",
        kind="agent",
        name="test",
        description="test",
        version="0.1.0",
    )
    return CompiledContract.from_spec(spec)


def _make_pre_event() -> PreAction:
    return PreAction(session_id="s1", contract_id="test", tool="llm_call", args={})


def _make_enforcer_with_ceiling(max_usd: float, action: str = "deny") -> SessionEnforcer:
    spec = ContractSpecExtended(
        dsl_version="0.4",
        contractspec="typec/v0.4",
        kind="agent",
        name="test",
        description="test",
        version="0.1.0",
        invariants=InvariantsExtended(
            process=ProcessInvariants(
                cost_ceiling=CostCeiling(max_usd_per_session=max_usd, action_on_breach=action)
            )
        ),
    )
    return SessionEnforcer(spec)


class _FakeCanonical:
    def __init__(self, provider: str) -> None:
        self.provider = provider


def test_no_ceiling_passes() -> None:
    compiled = _make_compiled_no_ceiling()
    result = evaluate_cost_ceiling(_make_pre_event(), compiled, 999.99, ViolationLog())
    assert result is None


def test_under_ceiling_passes() -> None:
    compiled = _make_compiled(max_usd=5.00, action="deny")
    result = evaluate_cost_ceiling(_make_pre_event(), compiled, 4.99, ViolationLog())
    assert result is None


def test_over_ceiling_deny() -> None:
    compiled = _make_compiled(max_usd=1.00, action="deny")
    violations = ViolationLog()
    result = evaluate_cost_ceiling(_make_pre_event(), compiled, 1.50, violations)
    assert result is not None
    assert result.decision == TypeCDecision.DENY
    assert "cost_ceiling" in result.violation_name
    v = violations.all_violations()
    assert len(v) == 1
    assert v[0]["kind"] == "hard"


def test_over_ceiling_warn() -> None:
    compiled = _make_compiled(max_usd=1.00, action="warn")
    violations = ViolationLog()
    result = evaluate_cost_ceiling(_make_pre_event(), compiled, 1.50, violations)
    assert result is None
    v = violations.all_violations()
    assert len(v) == 1
    assert v[0]["kind"] == "soft"


def test_extract_usage_anthropic() -> None:
    data = {"usage": {"input_tokens": 100, "output_tokens": 200}}
    assert extract_usage(data, "anthropic") == (100, 200)


def test_extract_usage_openai() -> None:
    data = {"usage": {"prompt_tokens": 150, "completion_tokens": 300}}
    assert extract_usage(data, "openai") == (150, 300)


def test_extract_usage_gemini() -> None:
    data = {"usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 120}}
    assert extract_usage(data, "gemini") == (80, 120)


def test_extract_usage_missing() -> None:
    data = {"content": "hello world"}
    assert extract_usage(data, "anthropic") is None


def test_cost_accumulates_across_requests() -> None:
    enforcer = _make_enforcer_with_ceiling(max_usd=10.00)
    resp1 = {"usage": {"input_tokens": 1000, "output_tokens": 500}}
    resp2 = {"usage": {"input_tokens": 2000, "output_tokens": 1000}}

    update_cost(resp1, _FakeCanonical("anthropic"), enforcer)
    update_cost(resp2, _FakeCanonical("anthropic"), enforcer)

    assert enforcer._accumulated_cost_usd > 0.0
    assert enforcer._accumulated_cost_usd < 1.0


def test_provider_price_map_overrides_default() -> None:
    enforcer = _make_enforcer_with_ceiling(max_usd=100.00)
    ceiling = CostCeiling(
        max_usd_per_session=100.00,
        action_on_breach="deny",
        provider_price_map={"anthropic": ProviderPriceEntry(input=100.0, output=500.0)},
    )
    enforcer._compiled.cost_ceiling_config = ceiling

    resp = {"usage": {"input_tokens": 1000, "output_tokens": 1000}}
    update_cost(resp, _FakeCanonical("anthropic"), enforcer)

    assert abs(enforcer._accumulated_cost_usd - 0.6) < 1e-6


def test_cost_persisted_to_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    enforcer = _make_enforcer_with_ceiling(max_usd=10.00)
    db_path = str(tmp_path / "cost-test.db")
    store = SessionStore(db_path)
    store.open()
    enforcer.attach_store(store)

    resp = {"usage": {"prompt_tokens": 500, "completion_tokens": 200}}
    update_cost(resp, _FakeCanonical("openai"), enforcer)

    store.flush()
    stored = store.get("cost")
    assert stored is not None
    assert "accumulated_usd" in stored
    assert stored["accumulated_usd"] > 0.0
    enforcer.close()


def test_cost_loaded_from_store_on_restart(tmp_path) -> None:  # type: ignore[no-untyped-def]
    db_path = str(tmp_path / "cost-restart.db")

    enforcer1 = _make_enforcer_with_ceiling(max_usd=10.00)
    store1 = SessionStore(db_path)
    store1.open()
    enforcer1.attach_store(store1)
    enforcer1._accumulated_cost_usd = 3.456
    enforcer1.close()

    enforcer2 = _make_enforcer_with_ceiling(max_usd=10.00)
    store2 = SessionStore(db_path)
    store2.open()
    enforcer2.attach_store(store2)

    assert abs(enforcer2._accumulated_cost_usd - 3.456) < 1e-6
    enforcer2.close()


def test_status_fields_shape() -> None:
    """Verify that the cost section fields are correctly shaped."""
    enforcer = _make_enforcer_with_ceiling(max_usd=5.00)
    enforcer._accumulated_cost_usd = 1.23

    ceiling_config = enforcer._compiled.cost_ceiling_config
    accumulated = enforcer._accumulated_cost_usd
    ceiling_usd = ceiling_config.max_usd_per_session if ceiling_config else None
    remaining = (ceiling_usd - accumulated) if ceiling_usd is not None else None

    assert ceiling_usd == 5.00
    assert remaining is not None
    assert abs(remaining - 3.77) < 1e-6
    assert accumulated == 1.23
