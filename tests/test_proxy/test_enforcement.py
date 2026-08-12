# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `packages/proxy/tests/test_enforcement.py`.

`SessionMonitor` -> `SessionEnforcer` (the migration notes). Fixtures reused from
`tests/test_gateway/fixtures/contracts/` rather than duplicated.
"""

from __future__ import annotations

from pathlib import Path

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.proxy.hot_reload import ContractWatcher
from agentassert_abc.proxy.server import create_app

FIXTURES = Path(__file__).parent.parent / "test_gateway" / "fixtures" / "contracts"


@pytest_asyncio.fixture
async def blocklist_client():
    app = create_app(str(FIXTURES / "safety-minimal.yaml"))
    enforcer = SessionEnforcer.from_yaml(str(FIXTURES / "safety-minimal.yaml"))
    app.state.monitor = enforcer
    app.state.upstream_overrides = None
    app.state.watcher = ContractWatcher(str(FIXTURES / "safety-minimal.yaml"))
    app.state.watcher.set_monitor(enforcer)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers={"accept-encoding": "identity"}
    ) as client:
        yield client
    enforcer.close()


class TestEnforcement:
    async def test_health_endpoint(self, blocklist_client) -> None:
        resp = await blocklist_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "contract" in data

    async def test_status_endpoint(self, blocklist_client) -> None:
        resp = await blocklist_client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "theta" in data
        assert "violations" in data


class TestHotReload:
    def test_watcher_no_change(self, tmp_path) -> None:
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            "dsl_version: '0.4'\ncontractspec: '1.0'\nkind: agent\n"
            "name: test\ndescription: test\nversion: '0.1'\n"
        )

        watcher = ContractWatcher(str(contract))
        watcher.set_monitor(None)
        watcher.start()
        import time

        time.sleep(0.7)
        result = watcher.swap_if_pending()
        assert result is None
        watcher.stop()

    def test_watcher_new_content(self, tmp_path) -> None:
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            "dsl_version: '0.4'\ncontractspec: '1.0'\nkind: agent\n"
            "name: test-a\ndescription: test\nversion: '0.1'\n"
        )

        watcher = ContractWatcher(str(contract))
        watcher.set_monitor(None)
        watcher.start()
        import time

        time.sleep(0.7)

        contract.write_text(
            "dsl_version: '0.4'\ncontractspec: '1.0'\nkind: agent\n"
            "name: test-b\ndescription: test\nversion: '0.1'\n"
        )
        time.sleep(0.7)

        result = watcher.swap_if_pending()
        assert result is not None
        watcher.stop()

    def test_watcher_invalid_contract(self, tmp_path) -> None:
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            "dsl_version: '0.4'\ncontractspec: '1.0'\nkind: agent\n"
            "name: test\ndescription: test\nversion: '0.1'\n"
        )

        watcher = ContractWatcher(str(contract))
        watcher.set_monitor(None)
        watcher.start()
        import time

        time.sleep(0.7)

        contract.write_text("invalid: : yaml")
        time.sleep(0.7)

        result = watcher.swap_if_pending()
        assert result is None
        watcher.stop()
