# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_persistence.py` — 15 tests minimum.

`test_serializer_drift_roundtrip` is REWRITTEN: typec's own `DriftTracker`
shape (`_call_sequence`, `_baseline_counts`, `_total_updates`) does not
exist in abc v2's `DriftTracker` (port-delta §C7). It now exercises abc's
actual shape (`history`, `_action_window`, `_reference`) via the gateway's
`serializers.dump_drift`/`load_drift`.

`test_db_isolation_by_session_id` (typec proxy's `_resolve_db_path`) is
dropped — the proxy subpackage (Phase E) is out of scope for this port.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading

import pytest

from agentassert_abc.gateway import serializers
from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.gateway.events import TurnEnd
from agentassert_abc.gateway.persistence import SessionStore
from agentassert_abc.gateway.violation_log import ViolationLog
from agentassert_abc.metrics.drift import DriftTracker
from agentassert_abc.metrics.theta import ThetaScorer
from agentassert_abc.models import DriftConfig
from agentassert_abc.process.models import ContractSpecExtended


@pytest.fixture
def db_path(tmp_path):  # type: ignore[no-untyped-def]
    return str(tmp_path / "test_session.db")


@pytest.fixture
def store(db_path):  # type: ignore[no-untyped-def]
    s = SessionStore(db_path)
    s.open()
    yield s
    with contextlib.suppress(Exception):
        s.close()


def _make_minimal_contract() -> ContractSpecExtended:
    return ContractSpecExtended(
        dsl_version="0.4",
        contractspec="typec/v0.4",
        kind="agent",
        name="test-contract",
        description="test",
        version="0.1.0",
    )


def _make_enforcer() -> SessionEnforcer:
    return SessionEnforcer(_make_minimal_contract())


def test_store_open_creates_db_file(db_path) -> None:  # type: ignore[no-untyped-def]
    s = SessionStore(db_path)
    assert not os.path.exists(db_path)
    s.open()
    assert os.path.exists(db_path)
    s.close()


def test_store_put_get_roundtrip(store) -> None:  # type: ignore[no-untyped-def]
    store.put("key1", {"hello": "world", "num": 42})
    store.flush()
    assert store.get("key1") == {"hello": "world", "num": 42}


def test_store_get_missing_key_returns_none(store) -> None:  # type: ignore[no-untyped-def]
    assert store.get("nonexistent_key") is None


def test_store_flush_writes_to_disk(db_path, store) -> None:  # type: ignore[no-untyped-def]
    store.put("check_key", {"value": 99})
    store.flush()

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT value FROM session_state WHERE key = ?", ("check_key",))
    row = cur.fetchone()
    conn.close()

    assert row is not None
    assert json.loads(row[0]) == {"value": 99}


def test_store_write_behind_not_immediate(db_path, store) -> None:  # type: ignore[no-untyped-def]
    store.flush()
    store.put("wb_key", {"x": 1})

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT value FROM session_state WHERE key = ?", ("wb_key",))
    row = cur.fetchone()
    conn.close()
    assert row is None


def test_store_close_flushes(db_path) -> None:  # type: ignore[no-untyped-def]
    s = SessionStore(db_path)
    s.open()
    s.put("close_key", {"data": "present"})
    s.close()

    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT value FROM session_state WHERE key = ?", ("close_key",))
    row = cur.fetchone()
    conn.close()
    assert row is not None
    assert json.loads(row[0]) == {"data": "present"}


def test_serializer_theta_roundtrip() -> None:
    theta = ThetaScorer()
    theta.record_compliance(0.9, 0.8)
    theta.record_drift(0.15)
    theta.record_violation()
    theta.apply_penalty(0.05)

    data = serializers.dump_theta(theta)

    theta2 = ThetaScorer()
    serializers.load_theta(theta2, data)

    assert theta2._compliance_scores == theta._compliance_scores
    assert theta2._drift_scores == theta._drift_scores
    assert theta2._violation_count == theta._violation_count
    assert abs(theta2._penalty_sum - theta._penalty_sum) < 1e-9


def test_serializer_drift_roundtrip() -> None:
    """Exercises abc v2's DriftTracker shape (history/_action_window/_reference)."""
    drift = DriftTracker(config=DriftConfig(window=10))
    drift.set_reference({"read_file": 0.5, "bash": 0.5})
    for tool in ["read_file", "bash", "read_file", "write_file", "bash"] * 2:
        drift.compute_drift(c_total=0.9, action_dist={tool: 1.0})

    data = serializers.dump_drift(drift)

    drift2 = DriftTracker(config=DriftConfig(window=10))
    serializers.load_drift(drift2, data)

    assert list(drift2.history) == list(drift.history)
    assert list(drift2._action_window) == list(drift._action_window)
    assert drift2._reference == drift._reference


def test_serializer_violations_roundtrip() -> None:
    log = ViolationLog()
    log.record("tool_blocklist", "PreAction", "bash", "blocked")
    log.record_soft("pii_filter", "PostAction", "response", "email found")

    data = serializers.dump_violations(log)
    assert len(data) == 2

    log2 = ViolationLog()
    serializers.load_violations(log2, data)

    violations = log2.all_violations()
    assert len(violations) == 2
    assert violations[0]["name"] == "tool_blocklist"
    assert violations[1]["name"] == "pii_filter"


def test_serializer_meta_roundtrip() -> None:
    enforcer = _make_enforcer()
    enforcer._turn_count = 7
    enforcer._deny_count = 3
    enforcer._seen_tools_session = {"bash", "read_file", "write_file"}

    data = serializers.dump_meta(enforcer)

    enforcer2 = _make_enforcer()
    serializers.load_meta(enforcer2, data)

    assert enforcer2._turn_count == 7
    assert enforcer2._deny_count == 3
    assert enforcer2._seen_tools_session == {"bash", "read_file", "write_file"}
    assert enforcer2._seen_tools_turn == set()


def test_enforcer_attach_store_loads_state(db_path) -> None:  # type: ignore[no-untyped-def]
    enforcer1 = _make_enforcer()
    store1 = SessionStore(db_path)
    store1.open()
    enforcer1.attach_store(store1)

    enforcer1._violations.record("tool_blocklist", "PreAction", "bash", "blocked")
    enforcer1._turn_count = 5
    enforcer1._deny_count = 2
    enforcer1._seen_tools_session.add("bash")
    enforcer1.close()

    enforcer2 = _make_enforcer()
    store2 = SessionStore(db_path)
    store2.open()
    enforcer2.attach_store(store2)

    assert enforcer2._turn_count == 5
    assert enforcer2._deny_count == 2
    assert "bash" in enforcer2._seen_tools_session
    violations = enforcer2._violations.all_violations()
    assert len(violations) == 1
    assert violations[0]["name"] == "tool_blocklist"

    enforcer2.close()


def test_enforcer_state_survives_restart(db_path) -> None:  # type: ignore[no-untyped-def]
    """THE critical test: full lifecycle survives a restart against the same DB."""
    enforcer1 = _make_enforcer()
    store1 = SessionStore(db_path)
    store1.open()
    enforcer1.attach_store(store1)

    enforcer1._violations.record("tool_blocklist", "PreAction", "bash", "bash is blocked")
    enforcer1._violations.record_soft("pii_filter", "PostAction", "response", "email found")
    enforcer1._turn_count = 12
    enforcer1._deny_count = 4
    enforcer1._theta.record_compliance(0.85, 0.75)
    enforcer1._theta.record_violation()

    enforcer1.close()

    enforcer2 = _make_enforcer()
    store2 = SessionStore(db_path)
    store2.open()
    enforcer2.attach_store(store2)

    assert enforcer2._turn_count == 12
    assert enforcer2._deny_count == 4
    violations = enforcer2._violations.all_violations()
    assert len(violations) == 2

    names = {v["name"] for v in violations}
    assert "tool_blocklist" in names
    assert "pii_filter" in names

    assert len(enforcer2._theta._compliance_scores) == 1
    assert enforcer2._theta._violation_count == 1

    enforcer2.close()


def test_enforcer_no_persist_mode() -> None:
    enforcer = _make_enforcer()
    enforcer._violations.record("tool_blocklist", "PreAction", "bash", "blocked")
    enforcer._turn_count = 3

    event = TurnEnd(session_id="s1", contract_id="test-contract", assistant_output="")
    result = enforcer.evaluate(event)
    assert result is not None

    se = enforcer.close()
    assert se is not None


def test_concurrent_evaluate_no_corruption(db_path) -> None:  # type: ignore[no-untyped-def]
    enforcer = _make_enforcer()
    store = SessionStore(db_path)
    store.open()
    enforcer.attach_store(store)

    errors: list[str] = []
    results: list[object] = []

    def worker() -> None:
        try:
            event = TurnEnd(session_id="s1", contract_id="test-contract", assistant_output="")
            results.append(enforcer.evaluate(event))
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == [], f"Errors during concurrent evaluate: {errors}"
    assert len(results) == 10
    enforcer.close()
