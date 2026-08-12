# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""FastAPI app factory — the AgentAssert enforcement proxy.

Ported from agentassert-typec's `server.py`. `SessionMonitor` ->
`SessionEnforcer` throughout. The `/status` endpoint's
drift section is built via `enforcer._build_drift_report()` — abc v2's
`DriftTracker` has no `report()` method; the enforcer's
own drift-report builder is the correct replacement.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.gateway.state import assert_evaluable_on_response_surface
from agentassert_abc.proxy import forwarder
from agentassert_abc.proxy.hot_reload import ContractWatcher
from agentassert_abc.proxy.routes import anthropic, gemini, openai, openrouter


def _extract_upstream(enforcer: SessionEnforcer) -> dict[str, str] | None:
    upstream = getattr(enforcer._contract, "upstream", None)
    if upstream is None:
        return None
    result = {}
    for provider in ("anthropic", "openai", "gemini", "openrouter"):
        url = getattr(upstream, provider, None)
        if url:
            result[provider] = url
    return result or None


def _resolve_db_path(contract_path: str, session_id: str | None = None) -> str:
    """Resolve the SQLite DB path for a given contract + optional session_id."""
    slug = re.sub(r"[^a-z0-9]+", "-", Path(contract_path).stem.lower()).strip("-")
    if session_id:
        slug = f"{slug}_{session_id}"
    db_dir = Path.home() / ".agentassert" / "sessions"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / f"{slug}.db")


def create_app(
    contract_path: str,
    session_id: str | None = None,
    persist: bool = True,
) -> FastAPI:
    enforcer_store: dict[str, SessionEnforcer] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            enforcer = SessionEnforcer.from_yaml(contract_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load contract: {e}") from e

        # Refuse to start on a contract this surface can never evaluate. The proxy
        # only observes the provider response, so an invariant over anything else
        # would score as a violation on every turn regardless of agent behaviour —
        # silently turning a compliant agent into a failing one.
        assert_evaluable_on_response_surface(enforcer._contract, "HTTP proxy")

        enforcer_store["enforcer"] = enforcer
        app.state.monitor = enforcer
        app.state.upstream_overrides = _extract_upstream(enforcer)
        app.state.db_path = None

        if persist:
            from agentassert_abc.gateway.persistence import SessionStore

            db_path = _resolve_db_path(contract_path, session_id)
            store = SessionStore(db_path)
            store.open()
            enforcer.attach_store(store)
            app.state.db_path = db_path
            app.state.store = store

        watcher = ContractWatcher(contract_path)
        watcher.set_monitor(enforcer)
        watcher.start()
        app.state.watcher = watcher

        yield

        if "enforcer" in enforcer_store:
            enforcer_store["enforcer"].close()  # flushes + closes store
        app.state.watcher.stop()
        if forwarder._client is not None:
            await forwarder._client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.contract_name = contract_path

    app.include_router(anthropic.router, prefix="/anthropic")
    app.include_router(openai.router, prefix="/openai")
    app.include_router(gemini.router, prefix="/gemini")
    app.include_router(openrouter.router, prefix="/openrouter")

    @app.get("/health")
    async def health(request: Request):
        enforcer: SessionEnforcer = request.app.state.monitor
        db_path = getattr(request.app.state, "db_path", None)
        store = getattr(request.app.state, "store", None)
        persistence = {
            "enabled": db_path is not None,
            "db_path": db_path,
            "dirty": store.is_dirty() if store is not None else False,
        }
        return {
            "status": "ok",
            "contract": enforcer._contract.name,
            "theta": enforcer._theta.compute(),
            "upstream": request.app.state.upstream_overrides or "defaults",
            "persistence": persistence,
        }

    @app.get("/status")
    async def status(request: Request):
        enforcer: SessionEnforcer = request.app.state.monitor
        drift_report = enforcer._build_drift_report()

        ceiling_config = enforcer._compiled.cost_ceiling_config
        with enforcer._cost_lock:
            accumulated = enforcer._accumulated_cost_usd
        ceiling_usd = ceiling_config.max_usd_per_session if ceiling_config else None
        remaining = (ceiling_usd - accumulated) if ceiling_usd is not None else None
        pct_used = (accumulated / ceiling_usd * 100.0) if ceiling_usd else None

        return {
            "theta": enforcer._theta.compute(),
            "drift": {
                "jsd": drift_report.current_jsd,
                "window": drift_report.window_size,
            },
            "violations": len(enforcer._violations.all_violations()),
            "cost": {
                "accumulated_usd": accumulated,
                "ceiling_usd": ceiling_usd,
                "remaining_usd": remaining,
                "pct_used": round(pct_used, 1) if pct_used is not None else None,
            },
        }

    @app.post("/admin/reload")
    async def admin_reload(request: Request):
        watcher = request.app.state.watcher
        new_enforcer = watcher.swap_if_pending()
        if new_enforcer:
            request.app.state.monitor = new_enforcer
            request.app.state.upstream_overrides = _extract_upstream(new_enforcer)
            return JSONResponse({"status": "reloaded", "contract": new_enforcer._contract.name})
        return JSONResponse({"status": "no_change"})

    return app
