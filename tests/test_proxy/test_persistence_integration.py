# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec
`packages/proxy/tests/test_persistence_integration.py`.

`SessionMonitor` -> `SessionEnforcer` (the migration notes). Note: `ThetaScorer`
now uses the abc v2 `(c_hard + c_soft) / 2` compliance formula (the migration notes
§C6, silent-break #1) rather than typec's `0.7*c_hard + 0.3*c_soft` — this
test only checks round-trip preservation across a restart, so it is
formula-agnostic and needs no value changes.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from httpx import ASGITransport, AsyncClient

from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.gateway.persistence import SessionStore
from agentassert_abc.proxy.hot_reload import ContractWatcher
from agentassert_abc.proxy.server import create_app

if TYPE_CHECKING:
    from pathlib import Path

_MINIMAL_CONTRACT = """
dsl_version: '0.4'
contractspec: 'agentassert-abc/v0.4'
kind: agent
name: test-persist
description: test
version: '0.1.0'
"""


def _write_contract(tmp_path: Path) -> Path:
    c = tmp_path / "test-persist.yaml"
    c.write_text(_MINIMAL_CONTRACT)
    return c


# ---------------------------------------------------------------------------
# test_proxy_restart_preserves_theta
# ---------------------------------------------------------------------------


def test_proxy_restart_preserves_theta(tmp_path) -> None:
    """Create enforcer, record theta state, close, reopen with same DB — theta is preserved."""
    contract = _write_contract(tmp_path)
    db_path = str(tmp_path / "theta-session.db")

    enforcer1 = SessionEnforcer.from_yaml(str(contract))
    store1 = SessionStore(db_path)
    store1.open()
    enforcer1.attach_store(store1)

    enforcer1._theta.record_compliance(0.9, 0.85)
    enforcer1._theta.record_compliance(0.8, 0.75)
    enforcer1._theta.record_drift(0.1)
    enforcer1._theta.apply_penalty(0.02)
    theta_before = enforcer1._theta.compute()
    enforcer1.close()

    enforcer2 = SessionEnforcer.from_yaml(str(contract))
    store2 = SessionStore(db_path)
    store2.open()
    enforcer2.attach_store(store2)

    theta_after = enforcer2._theta.compute()
    assert abs(theta_before - theta_after) < 1e-6, (
        f"Theta not preserved: before={theta_before:.6f}, after={theta_after:.6f}"
    )
    enforcer2.close()


# ---------------------------------------------------------------------------
# test_proxy_restart_preserves_violations
# ---------------------------------------------------------------------------


def test_proxy_restart_preserves_violations(tmp_path) -> None:
    """Violations accumulated before restart appear in the enforcer after restart."""
    contract = _write_contract(tmp_path)
    db_path = str(tmp_path / "violations-session.db")

    enforcer1 = SessionEnforcer.from_yaml(str(contract))
    store1 = SessionStore(db_path)
    store1.open()
    enforcer1.attach_store(store1)

    enforcer1._violations.record("tool_blocklist", "PreAction", "bash", "blocked")
    enforcer1._violations.record("tool_blocklist", "PreAction", "rm", "blocked")
    enforcer1._violations.record_soft("pii_filter", "PostAction", "response", "email found")
    enforcer1._turn_count = 10
    enforcer1._deny_count = 2
    enforcer1.close()

    enforcer2 = SessionEnforcer.from_yaml(str(contract))
    store2 = SessionStore(db_path)
    store2.open()
    enforcer2.attach_store(store2)

    violations = enforcer2._violations.all_violations()
    assert len(violations) == 3
    assert enforcer2._turn_count == 10
    assert enforcer2._deny_count == 2

    names = [v["name"] for v in violations]
    assert names.count("tool_blocklist") == 2
    assert names.count("pii_filter") == 1

    enforcer2.close()


# ---------------------------------------------------------------------------
# test_health_shows_persistence_info
# ---------------------------------------------------------------------------


def test_health_shows_persistence_info(tmp_path) -> None:
    """The /health endpoint returns persistence.enabled and persistence.db_path."""
    contract = _write_contract(tmp_path)
    db_path = str(tmp_path / "health-test.db")

    async def _run() -> None:
        app = create_app(str(contract), persist=False)  # no lifespan store
        enforcer = SessionEnforcer.from_yaml(str(contract))

        store = SessionStore(db_path)
        store.open()
        enforcer.attach_store(store)

        app.state.monitor = enforcer
        app.state.upstream_overrides = None
        app.state.db_path = db_path
        app.state.store = store
        app.state.watcher = ContractWatcher(str(contract))
        app.state.watcher.set_monitor(enforcer)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"accept-encoding": "identity"},
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "persistence" in data
            p = data["persistence"]
            assert p["enabled"] is True
            assert p["db_path"] is not None
            assert p["db_path"].endswith(".db")
            assert "dirty" in p

        enforcer.close()

    asyncio.run(_run())
