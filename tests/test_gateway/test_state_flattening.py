# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Response-surface state flattening and the contract-evaluability gate.

Regression for the defect where the HTTP proxy and the Claude Code hook never
populated constraint state: the proxy sent only a byte count and the hook sent
nothing, so every semantic invariant evaluated ``False`` and a fully compliant
agent was scored at ``c_hard = 0`` with a violation logged on every turn.
"""

from __future__ import annotations

import pytest

from agentassert_abc.dsl.parser import load_contract
from agentassert_abc.evaluator.engine import evaluate
from agentassert_abc.exceptions import ContractLoadError
from agentassert_abc.gateway.state import (
    assert_evaluable_on_response_surface,
    contract_field_names,
    flatten_output,
)

_CONTRACT = "tests/test_gateway/fixtures/contracts/abc-v03-compat.yaml"


def test_flatten_maps_response_body_to_dotted_output_keys() -> None:
    state = flatten_output({"pii_detected": False, "decision": "approve"})
    assert state["output.pii_detected"] is False
    assert state["output.decision"] == "approve"


def test_flatten_descends_into_nested_mappings() -> None:
    state = flatten_output({"usage": {"tokens": 42}})
    assert state["output.usage.tokens"] == 42
    # container retained so `exists` checks against it still resolve
    assert state["output.usage"] == {"tokens": 42}


def test_flatten_puts_unstructured_payloads_on_raw() -> None:
    assert flatten_output("plain text")["output.raw"] == "plain text"
    assert flatten_output(7)["output.raw"] == 7
    assert flatten_output(None) == {}


def test_flatten_never_raises_on_odd_payloads() -> None:
    class Weird:
        def __repr__(self) -> str:
            return "weird"

    assert flatten_output(Weird())["output.raw"] == "weird"
    assert flatten_output([1, 2, 3])["output.length"] == 3


def test_flatten_is_depth_bounded() -> None:
    deep: dict = {"a": {}}
    node = deep["a"]
    for _ in range(40):
        node["a"] = {}
        node = node["a"]
    flatten_output(deep)  # must terminate rather than recurse without bound


def test_flattened_response_makes_a_compliant_agent_score_compliant() -> None:
    """The actual regression: proxy-shaped state used to give c_hard = 0."""
    spec = load_contract(_CONTRACT)

    old_proxy_state = {"response_bytes": 4096}
    assert evaluate(spec, old_proxy_state).c_hard == 0.0  # the defect

    new_proxy_state = {"response.bytes": 4096}
    new_proxy_state.update(flatten_output({"pii_detected": False}))
    assert evaluate(spec, new_proxy_state).c_hard == 1.0  # fixed


def test_flattened_response_still_catches_a_real_violation() -> None:
    """The fix must not simply make everything pass."""
    spec = load_contract(_CONTRACT)
    state = flatten_output({"pii_detected": True})
    result = evaluate(spec, state)
    assert result.c_hard == 0.0
    assert len(result.hard_violations) == 1


def test_contract_field_names_lists_hard_and_soft_fields() -> None:
    fields = contract_field_names(load_contract(_CONTRACT))
    assert "output.pii_detected" in fields


def test_evaluability_gate_accepts_a_response_scoped_contract() -> None:
    assert_evaluable_on_response_surface(load_contract(_CONTRACT), "HTTP proxy")


def test_evaluability_gate_rejects_fields_the_surface_cannot_supply(tmp_path) -> None:
    """A contract over state the proxy never sees must fail at load, not silently."""
    contract = tmp_path / "unusable.yaml"
    contract.write_text(
        'dsl_version: "0.3"\n'
        'contractspec: "1.0"\n'
        "kind: agent\n"
        "name: unusable\n"
        'description: "references state the proxy never sees"\n'
        'version: "0.1"\n'
        "invariants:\n"
        "  hard:\n"
        "    - name: db-consistent\n"
        '      description: "internal state"\n'
        "      check:\n"
        "        field: database.rows_written\n"
        "        gt: 0\n"
    )
    spec = load_contract(str(contract))
    with pytest.raises(ContractLoadError, match="database.rows_written"):
        assert_evaluable_on_response_surface(spec, "HTTP proxy")
