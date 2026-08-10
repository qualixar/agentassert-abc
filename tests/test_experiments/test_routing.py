# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for RoutingClient — cross-backend dispatch by model id (LLD-F breadth).

RoutingClient enables the cross-backend sharing arms (different_vendor_meta /
different_vendor_grok) where a mission's two legs live on different providers.
All tests use injected fake transports — no real API calls, no network.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agentassert_abc.experiments import config


def _import_providers():
    from agentassert_abc.experiments import providers  # noqa: PLC0415

    return providers


def _chat_transport(tag_counter: dict, key: str):
    """OpenAI chat/completions-shaped fake transport that records a hit."""

    def _t(url: str, body: dict) -> dict:  # noqa: ARG001
        tag_counter[key] += 1
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "model": body.get("model", ""),
        }

    return _t


def _responses_transport(tag_counter: dict, key: str):
    """OpenAI Responses-shaped fake transport (Meta) that records a hit."""

    def _t(url: str, body: dict) -> dict:  # noqa: ARG001
        tag_counter[key] += 1
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "model": body.get("model", ""),
        }

    return _t


class TestRoutingClient:
    """RoutingClient dispatches each model id to the correct backend."""

    def test_routes_each_model_to_its_backend(self) -> None:
        p = _import_providers()
        hits = {"meta": 0, "grok": 0, "openrouter": 0}
        transports = {
            "meta": _responses_transport(hits, "meta"),  # Meta = Responses shape
            "grok": _chat_transport(hits, "grok"),
            "openrouter": _chat_transport(hits, "openrouter"),
        }
        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "k", "MODEL_API_KEY": "k"}),
        ):
            rc = p.RoutingClient(transports=transports)
            r_meta = rc.generate("muse-spark-1.2-contributor", "p")
            r_grok = rc.generate("grok-4.5", "p")
            rc.generate("mistralai/mistral-small-24b-instruct-2501", "p")
            rc.generate("google/gemma-3-12b-it", "p")

        # muse* → meta (1), grok* → grok (1), everything else → openrouter (2)
        assert hits == {"meta": 1, "grok": 1, "openrouter": 2}
        assert r_meta.text == "ok"
        assert r_grok.text == "ok"

    def test_lazy_subclients_only_build_what_is_used(self) -> None:
        """Routing only an OpenRouter model must not construct Meta/Grok."""
        p = _import_providers()
        hits = {"openrouter": 0}
        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}),  # NO MODEL_API_KEY
        ):
            rc = p.RoutingClient(
                transports={"openrouter": _chat_transport(hits, "openrouter")}
            )
            rc.generate("mistralai/mistral-small-24b-instruct-2501", "p")
        # If Meta had been constructed it would have raised on the missing key.
        assert hits["openrouter"] == 1

    def test_gate_closed_raises_at_construction(self) -> None:
        p = _import_providers()
        with (
            patch.object(config, "FRONTIER_ENABLED", False),
            pytest.raises(p.FrontierDisabledError),
        ):
            p.RoutingClient()
