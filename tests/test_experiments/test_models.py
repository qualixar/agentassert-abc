# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for agentassert_abc.experiments.models — LLD-E §4/§6.

TDD RED phase: written BEFORE the implementation.

Four critical invariants exercised here:
  1. SAFETY gate: FrontierClient.generate RAISES FrontierDisabledError when
     config.FRONTIER_ENABLED is False — the transport MUST NOT be invoked.
  2. LocalClient parses a mocked Ollama JSON body correctly; cost_usd == 0.0.
  3. FrontierClient enforces input and output token caps (FrontierTokenCapError).
  4. Frontier cost math: cost_usd = (in/1e6)*MAX_IN_PRICE + (out/1e6)*MAX_OUT_PRICE.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentassert_abc.experiments import config


# ---------------------------------------------------------------------------
# Deferred import so missing module fails on collection, not on first test
# ---------------------------------------------------------------------------


def _import_models():
    from agentassert_abc.experiments import models  # noqa: PLC0415

    return models


# ---------------------------------------------------------------------------
# ModelResponse contract
# ---------------------------------------------------------------------------


class TestModelResponse:
    def test_frozen_rejects_mutation(self) -> None:
        m = _import_models()
        resp = m.ModelResponse(
            text="hi",
            input_tokens=5,
            output_tokens=3,
            model="qwen2.5:7b",
            cost_usd=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            resp.text = "mutated"  # type: ignore[misc]

    def test_slots_present(self) -> None:
        """__slots__ means no per-instance __dict__."""
        m = _import_models()
        resp = m.ModelResponse(
            text="x",
            input_tokens=1,
            output_tokens=1,
            model="m",
            cost_usd=0.0,
        )
        assert not hasattr(resp, "__dict__")

    def test_fields_accessible(self) -> None:
        m = _import_models()
        resp = m.ModelResponse(
            text="result",
            input_tokens=10,
            output_tokens=20,
            model="qwen2.5:7b",
            cost_usd=0.0015,
        )
        assert resp.text == "result"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 20
        assert resp.model == "qwen2.5:7b"
        assert resp.cost_usd == pytest.approx(0.0015)


# ---------------------------------------------------------------------------
# LocalClient
# ---------------------------------------------------------------------------


def _ollama_payload(
    text: str = "result",
    input_tokens: int = 10,
    output_tokens: int = 20,
    model: str = "qwen2.5:7b",
) -> dict:
    return {
        "model": model,
        "response": text,
        "done": True,
        "prompt_eval_count": input_tokens,
        "eval_count": output_tokens,
    }


class TestLocalClient:
    def test_parses_mocked_response(self) -> None:
        m = _import_models()
        payload = _ollama_payload(text="42", input_tokens=15, output_tokens=7)

        def fake_transport(url: str, body: dict) -> dict:
            return payload

        client = m.LocalClient(transport=fake_transport)
        resp = client.generate(model="qwen2.5:7b", prompt="what is 6×7?")

        assert resp.text == "42"
        assert resp.input_tokens == 15
        assert resp.output_tokens == 7
        assert resp.model == "qwen2.5:7b"

    def test_cost_is_always_zero(self) -> None:
        m = _import_models()

        def fake_transport(url: str, body: dict) -> dict:
            return _ollama_payload(input_tokens=9999, output_tokens=8888)

        client = m.LocalClient(transport=fake_transport)
        resp = client.generate(model="qwen2.5:7b", prompt="test")
        assert resp.cost_usd == 0.0

    def test_transport_receives_correct_url_and_stream_false(self) -> None:
        m = _import_models()
        recorded: list[tuple[str, dict]] = []

        def recording_transport(url: str, body: dict) -> dict:
            recorded.append((url, body))
            return _ollama_payload()

        client = m.LocalClient(transport=recording_transport)
        client.generate(model="qwen2.5:7b", prompt="hello")

        assert len(recorded) == 1
        url, body = recorded[0]
        assert "/api/generate" in url
        assert body["stream"] is False
        assert body["model"] == "qwen2.5:7b"
        assert body["prompt"] == "hello"

    def test_transport_error_propagates(self) -> None:
        m = _import_models()

        def bad_transport(url: str, body: dict) -> dict:
            raise ConnectionError("no ollama")

        client = m.LocalClient(transport=bad_transport)
        with pytest.raises(ConnectionError):
            client.generate(model="qwen2.5:7b", prompt="test")

    def test_missing_response_key_raises_key_error(self) -> None:
        """Malformed Ollama payload without 'response' → KeyError, not silent None."""
        m = _import_models()

        def bad_transport(url: str, body: dict) -> dict:
            return {"model": "qwen2.5:7b", "done": True}  # 'response' missing

        client = m.LocalClient(transport=bad_transport)
        with pytest.raises(KeyError):
            client.generate(model="qwen2.5:7b", prompt="test")


# ---------------------------------------------------------------------------
# FrontierClient — SAFETY GATE (most critical)
# ---------------------------------------------------------------------------


class TestFrontierSafetyGate:
    def test_raises_frontier_disabled_error_when_disabled(self) -> None:
        """SAFETY: transport MUST NOT be called; exception must fire first."""
        m = _import_models()
        transport_calls: list[bool] = []

        def should_never_be_called(url: str, body: dict) -> dict:
            transport_calls.append(True)
            return {}

        with patch.object(config, "FRONTIER_ENABLED", False):
            client = m.FrontierClient(transport=should_never_be_called)
            with pytest.raises(m.FrontierDisabledError):
                client.generate(model="openai/gpt-5-mini", prompt="test")

        assert transport_calls == [], (
            "transport was invoked despite FRONTIER_ENABLED=False — safety gate broken"
        )

    def test_frontier_disabled_error_subclasses_agent_assert_error(self) -> None:
        from agentassert_abc.exceptions import AgentAssertError

        m = _import_models()
        assert issubclass(m.FrontierDisabledError, AgentAssertError)

    def test_frontier_disabled_error_message_mentions_flag(self) -> None:
        """Error message should guide the reader to the safety flag."""
        m = _import_models()
        with patch.object(config, "FRONTIER_ENABLED", False):
            client = m.FrontierClient(transport=lambda u, b: {})
            with pytest.raises(m.FrontierDisabledError, match="FRONTIER_ENABLED"):
                client.generate(model="openai/gpt-5-mini", prompt="test")

    def test_frontier_token_cap_error_subclasses_agent_assert_error(self) -> None:
        from agentassert_abc.exceptions import AgentAssertError

        m = _import_models()
        assert issubclass(m.FrontierTokenCapError, AgentAssertError)


# ---------------------------------------------------------------------------
# FrontierClient — token cap enforcement
# ---------------------------------------------------------------------------


def _frontier_payload(
    text: str = "answer",
    input_tokens: int = 100,
    output_tokens: int = 50,
    model: str = "openai/gpt-5-mini",
) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        },
        "model": model,
    }


class TestFrontierTokenCaps:
    def test_raises_on_input_tokens_exceeded(self) -> None:
        m = _import_models()
        over = config.FRONTIER_MAX_INPUT_TOKENS + 1

        def transport(url: str, body: dict) -> dict:
            return _frontier_payload(input_tokens=over, output_tokens=10)

        with patch.object(config, "FRONTIER_ENABLED", True):
            client = m.FrontierClient(transport=transport)
            with pytest.raises(m.FrontierTokenCapError):
                client.generate(model="openai/gpt-5-mini", prompt="test")

    def test_raises_on_output_tokens_exceeded(self) -> None:
        m = _import_models()
        over = config.FRONTIER_MAX_OUTPUT_TOKENS + 1

        def transport(url: str, body: dict) -> dict:
            return _frontier_payload(input_tokens=10, output_tokens=over)

        with patch.object(config, "FRONTIER_ENABLED", True):
            client = m.FrontierClient(transport=transport)
            with pytest.raises(m.FrontierTokenCapError):
                client.generate(model="openai/gpt-5-mini", prompt="test")

    def test_passes_at_exact_caps(self) -> None:
        """Calls at exactly the cap limits must succeed."""
        m = _import_models()

        def transport(url: str, body: dict) -> dict:
            return _frontier_payload(
                input_tokens=config.FRONTIER_MAX_INPUT_TOKENS,
                output_tokens=config.FRONTIER_MAX_OUTPUT_TOKENS,
            )

        with patch.object(config, "FRONTIER_ENABLED", True):
            client = m.FrontierClient(transport=transport)
            resp = client.generate(model="openai/gpt-5-mini", prompt="test")

        assert resp.input_tokens == config.FRONTIER_MAX_INPUT_TOKENS
        assert resp.output_tokens == config.FRONTIER_MAX_OUTPUT_TOKENS

    def test_token_cap_error_message_includes_counts(self) -> None:
        m = _import_models()
        over = config.FRONTIER_MAX_INPUT_TOKENS + 50

        def transport(url: str, body: dict) -> dict:
            return _frontier_payload(input_tokens=over, output_tokens=5)

        with patch.object(config, "FRONTIER_ENABLED", True):
            client = m.FrontierClient(transport=transport)
            with pytest.raises(m.FrontierTokenCapError, match=str(over)):
                client.generate(model="openai/gpt-5-mini", prompt="test")


# ---------------------------------------------------------------------------
# FrontierClient — cost math (LLD-E §6.2)
# ---------------------------------------------------------------------------


class TestFrontierCostMath:
    def test_cost_exact_formula(self) -> None:
        """cost_usd = (input/1e6)*MAX_IN_PRICE + (output/1e6)*MAX_OUT_PRICE."""
        m = _import_models()
        in_tok, out_tok = 400, 80
        expected = (
            in_tok / 1e6 * config.MAX_INPUT_PRICE_PER_M_USD
            + out_tok / 1e6 * config.MAX_OUTPUT_PRICE_PER_M_USD
        )

        def transport(url: str, body: dict) -> dict:
            return _frontier_payload(input_tokens=in_tok, output_tokens=out_tok)

        with patch.object(config, "FRONTIER_ENABLED", True):
            resp = m.FrontierClient(transport=transport).generate(
                model="openai/gpt-5-mini", prompt="test"
            )

        assert resp.cost_usd == pytest.approx(expected)

    def test_cost_at_ceiling_equals_per_call_ceiling(self) -> None:
        """At maximum admitted tokens, cost == PER_CALL_CEILING_USD."""
        m = _import_models()
        in_tok = config.FRONTIER_MAX_INPUT_TOKENS
        out_tok = config.FRONTIER_MAX_OUTPUT_TOKENS

        def transport(url: str, body: dict) -> dict:
            return _frontier_payload(input_tokens=in_tok, output_tokens=out_tok)

        with patch.object(config, "FRONTIER_ENABLED", True):
            resp = m.FrontierClient(transport=transport).generate(
                model="openai/gpt-5-mini", prompt="test"
            )

        assert resp.cost_usd == pytest.approx(config.PER_CALL_CEILING_USD)
        # Must never exceed ceiling (floating-point tolerance)
        assert resp.cost_usd <= config.PER_CALL_CEILING_USD + 1e-12

    def test_cost_zero_tokens(self) -> None:
        """Zero tokens → cost == 0."""
        m = _import_models()

        def transport(url: str, body: dict) -> dict:
            return _frontier_payload(input_tokens=0, output_tokens=0)

        with patch.object(config, "FRONTIER_ENABLED", True):
            resp = m.FrontierClient(transport=transport).generate(
                model="openai/gpt-5-mini", prompt="test"
            )

        assert resp.cost_usd == 0.0

    def test_response_fields_populated(self) -> None:
        """Verify text, model, and token fields land on ModelResponse."""
        m = _import_models()

        def transport(url: str, body: dict) -> dict:
            return _frontier_payload(
                text="The answer is 42",
                input_tokens=50,
                output_tokens=10,
                model="openai/gpt-5-mini",
            )

        with patch.object(config, "FRONTIER_ENABLED", True):
            resp = m.FrontierClient(transport=transport).generate(
                model="openai/gpt-5-mini", prompt="test"
            )

        assert resp.text == "The answer is 42"
        assert resp.input_tokens == 50
        assert resp.output_tokens == 10


# ---------------------------------------------------------------------------
# generate() dispatcher
# ---------------------------------------------------------------------------


class TestGenerateDispatcher:
    def test_routes_local_model_to_local_client(self) -> None:
        m = _import_models()
        payload = _ollama_payload(text="local answer", model="qwen2.5:7b")

        def transport(url: str, body: dict) -> dict:
            return payload

        resp = m.generate(model="qwen2.5:7b", prompt="test", transport=transport)
        assert resp.text == "local answer"
        assert resp.cost_usd == 0.0

    def test_routes_gemma_to_local_client(self) -> None:
        m = _import_models()

        def transport(url: str, body: dict) -> dict:
            return _ollama_payload(text="gemma answer", model="gemma3:4b")

        resp = m.generate(model="gemma3:4b", prompt="test", transport=transport)
        assert resp.cost_usd == 0.0

    def test_routes_frontier_model_raises_when_disabled(self) -> None:
        m = _import_models()
        with patch.object(config, "FRONTIER_ENABLED", False):
            with pytest.raises(m.FrontierDisabledError):
                m.generate(model="openai/gpt-5-mini", prompt="test")

    def test_routes_frontier_model_when_enabled(self) -> None:
        m = _import_models()

        def transport(url: str, body: dict) -> dict:
            return _frontier_payload(text="frontier", input_tokens=50, output_tokens=30)

        with patch.object(config, "FRONTIER_ENABLED", True):
            resp = m.generate(
                model="openai/gpt-5-mini", prompt="test", transport=transport
            )

        assert resp.text == "frontier"
        assert resp.cost_usd > 0.0
