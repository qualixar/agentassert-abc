# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_judge_http.py` +
`tests/test_otel_judge.py::TestJudgeDispatcher` (OTel-specific tests dropped
— see `tests/test_gateway/__init__.py`).

IP note: `model="ds-flash-free"` fixtures genericized to
`FREE_TIER_MODEL_ALIAS` per the port IP policy (port-delta §E.3).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentassert_abc.gateway.judge import FREE_TIER_MODEL_ALIAS, JudgeDispatcher


def _make_mock_response(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_data
    return resp


def _mock_async_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestJudgeDispatcherSampling:
    def test_should_sample_always_true_at_1(self) -> None:
        dispatcher = JudgeDispatcher(cost_ceiling=100.0)
        assert dispatcher.should_sample(1.0) is True

    def test_should_sample_always_false_at_0(self) -> None:
        dispatcher = JudgeDispatcher(cost_ceiling=100.0)
        assert dispatcher.should_sample(0.0) is False

    def test_should_sample_false_when_ceiling_exceeded(self) -> None:
        dispatcher = JudgeDispatcher(cost_ceiling=0.0)
        assert dispatcher.should_sample(1.0) is False

    def test_stats(self) -> None:
        dispatcher = JudgeDispatcher(cost_ceiling=10.0, model="haiku")
        assert dispatcher.stats["ceiling"] == 10.0
        assert dispatcher.total_spent == 0.0


class TestCallAnthropicHaiku:
    @pytest.mark.asyncio
    async def test_pass_verdict(self) -> None:
        resp = _make_mock_response(
            {
                "content": [{"type": "text", "text": "PASS"}],
                "usage": {"input_tokens": 100, "output_tokens": 5},
            }
        )
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            with patch("httpx.AsyncClient", return_value=_mock_async_client(resp)):
                d = JudgeDispatcher(model="haiku")
                result, cost = await d._call_anthropic_haiku("test prompt")
            assert result is True
            assert cost > 0.0
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    @pytest.mark.asyncio
    async def test_fail_verdict(self) -> None:
        resp = _make_mock_response(
            {
                "content": [{"type": "text", "text": "FAIL — did not meet rubric"}],
                "usage": {"input_tokens": 120, "output_tokens": 8},
            }
        )
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            with patch("httpx.AsyncClient", return_value=_mock_async_client(resp)):
                d = JudgeDispatcher(model="haiku")
                result, cost = await d._call_anthropic_haiku("test prompt")
            assert result is False
            assert cost > 0.0
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    @pytest.mark.asyncio
    async def test_no_content_blocks(self) -> None:
        resp = _make_mock_response(
            {"content": [], "usage": {"input_tokens": 50, "output_tokens": 2}}
        )
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            with patch("httpx.AsyncClient", return_value=_mock_async_client(resp)):
                d = JudgeDispatcher(model="haiku")
                result, _cost = await d._call_anthropic_haiku("test prompt")
            assert result is False
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    @pytest.mark.asyncio
    async def test_non_text_block_skipped(self) -> None:
        resp = _make_mock_response(
            {
                "content": [
                    {"type": "tool_use", "name": "compute"},
                    {"type": "text", "text": "PASS"},
                ],
                "usage": {"input_tokens": 80, "output_tokens": 4},
            }
        )
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            with patch("httpx.AsyncClient", return_value=_mock_async_client(resp)):
                d = JudgeDispatcher(model="haiku")
                result, _cost = await d._call_anthropic_haiku("test prompt")
            assert result is True
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    @pytest.mark.asyncio
    async def test_missing_usage_falls_back_to_estimate(self) -> None:
        resp = _make_mock_response({"content": [{"type": "text", "text": "PASS"}]})
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            with patch("httpx.AsyncClient", return_value=_mock_async_client(resp)):
                d = JudgeDispatcher(model="haiku")
                result, cost = await d._call_anthropic_haiku("a" * 400)
            assert result is True
            assert cost > 0.0
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    @pytest.mark.asyncio
    async def test_network_error_returns_fail_safe(self) -> None:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=ConnectionError("network"))
        cm.__aexit__ = AsyncMock(return_value=False)
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            with patch("httpx.AsyncClient", return_value=cm):
                d = JudgeDispatcher(model="haiku")
                result, cost = await d._call_anthropic_haiku("test prompt")
            assert result is True
            assert cost == 0.0
        finally:
            del os.environ["ANTHROPIC_API_KEY"]


class TestCallOpenRouterFree:
    @pytest.mark.asyncio
    async def test_pass_verdict(self) -> None:
        resp = _make_mock_response({"choices": [{"message": {"content": "PASS"}}]})
        os.environ["OPENROUTER_API_KEY"] = "or-test-key"
        try:
            with patch("httpx.AsyncClient", return_value=_mock_async_client(resp)):
                d = JudgeDispatcher(model=FREE_TIER_MODEL_ALIAS)
                result, cost = await d._call_openrouter_free("test prompt")
            assert result is True
            assert cost == 0.0
        finally:
            del os.environ["OPENROUTER_API_KEY"]

    @pytest.mark.asyncio
    async def test_fail_verdict(self) -> None:
        resp = _make_mock_response({"choices": [{"message": {"content": "FAIL"}}]})
        os.environ["OPENROUTER_API_KEY"] = "or-test-key"
        try:
            with patch("httpx.AsyncClient", return_value=_mock_async_client(resp)):
                d = JudgeDispatcher(model=FREE_TIER_MODEL_ALIAS)
                result, cost = await d._call_openrouter_free("test prompt")
            assert result is False
            assert cost == 0.0
        finally:
            del os.environ["OPENROUTER_API_KEY"]

    @pytest.mark.asyncio
    async def test_empty_choices_triggers_failsafe(self) -> None:
        resp = _make_mock_response({"choices": []})
        os.environ["OPENROUTER_API_KEY"] = "or-test-key"
        try:
            with patch("httpx.AsyncClient", return_value=_mock_async_client(resp)):
                d = JudgeDispatcher(model="free")
                result, cost = await d._call_openrouter_free("test prompt")
            assert result is True
            assert cost == 0.0
        finally:
            del os.environ["OPENROUTER_API_KEY"]

    @pytest.mark.asyncio
    async def test_network_error_returns_fail_safe(self) -> None:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=OSError("network"))
        cm.__aexit__ = AsyncMock(return_value=False)
        os.environ["OPENROUTER_API_KEY"] = "or-test-key"
        try:
            with patch("httpx.AsyncClient", return_value=cm):
                d = JudgeDispatcher(model=FREE_TIER_MODEL_ALIAS)
                result, cost = await d._call_openrouter_free("test prompt")
            assert result is True
            assert cost == 0.0
        finally:
            del os.environ["OPENROUTER_API_KEY"]


class TestEvaluateFullPath:
    @pytest.mark.asyncio
    async def test_evaluate_fail_increments_failure_count(self) -> None:
        resp = _make_mock_response(
            {
                "content": [{"type": "text", "text": "FAIL"}],
                "usage": {"input_tokens": 50, "output_tokens": 3},
            }
        )
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            with patch("httpx.AsyncClient", return_value=_mock_async_client(resp)):
                d = JudgeDispatcher(model="haiku", cost_ceiling=10.0)
                result, cost = await d.evaluate("rubric", "content", "s1")
            assert result is False
            assert d._failure_count == 1
            assert d._spent_usd > 0.0
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    @pytest.mark.asyncio
    async def test_evaluate_pass_no_failure_count(self) -> None:
        resp = _make_mock_response(
            {
                "content": [{"type": "text", "text": "PASS"}],
                "usage": {"input_tokens": 50, "output_tokens": 3},
            }
        )
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            with patch("httpx.AsyncClient", return_value=_mock_async_client(resp)):
                d = JudgeDispatcher(model="haiku", cost_ceiling=10.0)
                result, _cost = await d.evaluate("rubric", "content", "s1")
            assert result is True
            assert d._failure_count == 0
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    @pytest.mark.asyncio
    async def test_evaluate_exception_in_call_returns_fail_safe(self) -> None:
        d = JudgeDispatcher(model="haiku", cost_ceiling=10.0)
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            with patch.object(d, "_call_anthropic_haiku", side_effect=RuntimeError("boom")):
                result, cost = await d.evaluate("rubric", "content", "s1")
            assert result is True
            assert cost == 0.0
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    @pytest.mark.asyncio
    async def test_evaluate_no_api_key_haiku_passes(self) -> None:
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            dispatcher = JudgeDispatcher(model="haiku")
            passed, cost = await dispatcher.evaluate("rubric", "content", "s1")
            assert passed is True
            assert cost == 0.0
        finally:
            if saved:
                os.environ["ANTHROPIC_API_KEY"] = saved

    @pytest.mark.asyncio
    async def test_evaluate_free_model_no_key_passes(self) -> None:
        saved = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            dispatcher = JudgeDispatcher(model=FREE_TIER_MODEL_ALIAS)
            passed, cost = await dispatcher.evaluate("rubric", "content", "s1")
            assert passed is True
            assert cost == 0.0
        finally:
            if saved:
                os.environ["OPENROUTER_API_KEY"] = saved
