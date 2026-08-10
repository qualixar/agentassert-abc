# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for agentassert_abc.experiments.providers — Task #20 (LLD-E §4.1).

TDD RED phase: written BEFORE the implementation.  These tests define the
complete contract for the three frontier provider adapters.

Four critical invariant groups exercised:

(a) TestFrontierGateDisabled
    All three adapters MUST raise FrontierDisabledError when FRONTIER_ENABLED
    is False — at construction time, before any network activity.  The injected
    mock transport must never be called.

(b) TestHappyPath
    With FRONTIER_ENABLED patched True, a valid fake key, and a mocked 200
    response: sampling params are sent, cost_usd is computed from
    config.PROVIDER_PRICES, and ModelResponse is fully populated.

(c) TestRetryOnce
    A mocked 429 (first call) then 200 (second call) → one retry succeeds.
    The transport is called exactly twice.

(d) TestRetryExhausted
    A mocked persistent 5xx → raises the original RuntimeError after one retry.
    No further attempts beyond two total transport calls.

NO real network sockets are opened: every test injects an in-process mock
transport.  $0 spend guaranteed.

SAFETY ASSERTION: FRONTIER_ENABLED is never set True outside a combined
`with patch.object(...)` context manager.  No test persists a True state
after the `with` block exits.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agentassert_abc.experiments import config

# ---------------------------------------------------------------------------
# Lazy imports — tests fail on collection if module is missing (expected RED)
# ---------------------------------------------------------------------------


def _import_providers():
    from agentassert_abc.experiments import providers  # noqa: PLC0415

    return providers


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _chat_completions_payload(
    text: str = "answer",
    input_tokens: int = 100,
    output_tokens: int = 50,
    model: str = "qwen/qwen3-7b-fast",
) -> dict:
    """Build a minimal OpenAI chat/completions-compatible mock response."""
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        },
        "model": model,
    }


def _never_called_transport(url: str, body: dict) -> dict:  # noqa: ARG001
    """Transport that asserts it is never invoked."""
    pytest.fail(
        f"transport was called despite FRONTIER_ENABLED=False — "
        f"safety gate broken. URL={url!r}"
    )
    return {}  # unreachable; keeps type-checker happy


# ---------------------------------------------------------------------------
# (a) TestFrontierGateDisabled
#
# All three adapters must raise FrontierDisabledError at construction time
# when config.FRONTIER_ENABLED is False.  The mock transport must not be
# invoked — the gate fires first, before any I/O.
# ---------------------------------------------------------------------------


class TestFrontierGateDisabled:
    """Safety gate: FRONTIER_ENABLED=False → all adapters raise at __init__."""

    def test_openrouter_raises_when_disabled(self) -> None:
        p = _import_providers()
        with (
            patch.object(config, "FRONTIER_ENABLED", False),
            pytest.raises(p.FrontierDisabledError),
        ):
            p.OpenRouterClient(transport=_never_called_transport)

    def test_metaspark_raises_when_disabled(self) -> None:
        p = _import_providers()
        with (
            patch.object(config, "FRONTIER_ENABLED", False),
            pytest.raises(p.FrontierDisabledError),
        ):
            p.MetaSparkClient(transport=_never_called_transport)

    def test_grok_raises_when_disabled(self) -> None:
        p = _import_providers()
        with (
            patch.object(config, "FRONTIER_ENABLED", False),
            pytest.raises(p.FrontierDisabledError),
        ):
            p.GrokBridgeClient(transport=_never_called_transport)

    def test_openrouter_transport_never_called(self) -> None:
        """Explicitly assert the mock transport received zero calls."""
        p = _import_providers()
        transport_calls: list[str] = []

        def recording_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            transport_calls.append(url)
            return {}

        with (
            patch.object(config, "FRONTIER_ENABLED", False),
            pytest.raises(p.FrontierDisabledError),
        ):
            p.OpenRouterClient(transport=recording_transport)

        assert transport_calls == [], (
            "transport was invoked despite FRONTIER_ENABLED=False — "
            f"calls: {transport_calls}"
        )

    def test_metaspark_transport_never_called(self) -> None:
        p = _import_providers()
        transport_calls: list[str] = []

        def recording_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            transport_calls.append(url)
            return {}

        with (
            patch.object(config, "FRONTIER_ENABLED", False),
            pytest.raises(p.FrontierDisabledError),
        ):
            p.MetaSparkClient(transport=recording_transport)

        assert transport_calls == [], (
            f"transport was invoked (FRONTIER_ENABLED=False). calls: {transport_calls}"
        )

    def test_grok_transport_never_called(self) -> None:
        p = _import_providers()
        transport_calls: list[str] = []

        def recording_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            transport_calls.append(url)
            return {}

        with (
            patch.object(config, "FRONTIER_ENABLED", False),
            pytest.raises(p.FrontierDisabledError),
        ):
            p.GrokBridgeClient(transport=recording_transport)

        assert transport_calls == [], (
            f"transport was invoked (FRONTIER_ENABLED=False). calls: {transport_calls}"
        )

    def test_error_subclasses_agent_assert_error(self) -> None:
        from agentassert_abc.exceptions import AgentAssertError

        p = _import_providers()
        assert issubclass(p.FrontierDisabledError, AgentAssertError)

    def test_error_message_mentions_frontier_enabled(self) -> None:
        """The error message must guide the reader to the safety flag."""
        p = _import_providers()
        with (
            patch.object(config, "FRONTIER_ENABLED", False),
            pytest.raises(p.FrontierDisabledError, match="FRONTIER_ENABLED"),
        ):
            p.OpenRouterClient(transport=_never_called_transport)

    def test_openrouter_missing_key_also_raises(self) -> None:
        """If FRONTIER_ENABLED=True but key absent, also raises FrontierDisabledError."""
        p = _import_providers()
        # Ensure key is NOT set.
        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, env_without_key, clear=True),
            pytest.raises(p.FrontierDisabledError),
        ):
            p.OpenRouterClient(transport=_never_called_transport)

    def test_metaspark_missing_key_also_raises(self) -> None:
        """If FRONTIER_ENABLED=True but MODEL_API_KEY absent, raises FrontierDisabledError."""
        p = _import_providers()
        env_without_key = {k: v for k, v in os.environ.items() if k != "MODEL_API_KEY"}
        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, env_without_key, clear=True),
            pytest.raises(p.FrontierDisabledError),
        ):
            p.MetaSparkClient(transport=_never_called_transport)


# ---------------------------------------------------------------------------
# (b) TestHappyPath
#
# FRONTIER_ENABLED patched True + fake API key + mocked 200 response.
# Verifies: caps sent in body, cost_usd from PROVIDER_PRICES, well-formed
# ModelResponse.
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Happy path: enabled + key present + mocked 200 → correct ModelResponse."""

    def test_openrouter_returns_well_formed_response(self) -> None:
        p = _import_providers()
        payload = _chat_completions_payload(
            text="The answer is 4.", input_tokens=80, output_tokens=30
        )

        def mock_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            return payload

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
        ):
            client = p.OpenRouterClient(transport=mock_transport)
            resp = client.generate("qwen/qwen3-7b-fast", "What is 2+2?")

        assert resp.text == "The answer is 4."
        assert resp.input_tokens == 80
        assert resp.output_tokens == 30
        assert resp.model == "qwen/qwen3-7b-fast"
        assert isinstance(resp.cost_usd, float)

    def test_metaspark_returns_well_formed_response(self) -> None:
        p = _import_providers()
        payload = _chat_completions_payload(
            text="Meta response.", input_tokens=60, output_tokens=20
        )

        def mock_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            return payload

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"MODEL_API_KEY": "test-key-meta"}),
        ):
            client = p.MetaSparkClient(transport=mock_transport)
            resp = client.generate(config.META_CONTRIBUTOR_MODEL, "Classify.")

        assert resp.text == "Meta response."
        assert resp.input_tokens == 60
        assert resp.output_tokens == 20

    def test_grok_returns_well_formed_response(self) -> None:
        """GrokBridgeClient: no API key required, local proxy."""
        p = _import_providers()
        payload = _chat_completions_payload(
            text="Grok says hi.", input_tokens=50, output_tokens=15
        )

        def mock_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            return payload

        with patch.object(config, "FRONTIER_ENABLED", True):
            client = p.GrokBridgeClient(transport=mock_transport)
            resp = client.generate("grok-1.5-flash", "Hello?")

        assert resp.text == "Grok says hi."

    def test_sampling_params_in_request_body_openrouter(self) -> None:
        """Adapter must send temperature=0.2, top_p=1.0, max_tokens=FRONTIER_MAX_OUTPUT."""
        p = _import_providers()
        captured_bodies: list[dict] = []

        def capturing_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            captured_bodies.append(body)
            return _chat_completions_payload(input_tokens=50, output_tokens=20)

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
        ):
            client = p.OpenRouterClient(transport=capturing_transport)
            client.generate("qwen/qwen3-7b-fast", "test prompt")

        assert len(captured_bodies) == 1
        body = captured_bodies[0]
        assert body["max_tokens"] == config.FRONTIER_MAX_OUTPUT_TOKENS, (
            f"max_tokens should be {config.FRONTIER_MAX_OUTPUT_TOKENS}, "
            f"got {body.get('max_tokens')}"
        )
        assert body["temperature"] == pytest.approx(0.2)
        assert body["top_p"] == pytest.approx(1.0)

    def test_sampling_params_in_request_body_metaspark(self) -> None:
        p = _import_providers()
        captured_bodies: list[dict] = []

        def capturing_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            captured_bodies.append(body)
            return _chat_completions_payload(input_tokens=30, output_tokens=10)

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"MODEL_API_KEY": "test-key-meta"}),
        ):
            client = p.MetaSparkClient(transport=capturing_transport)
            client.generate(config.META_CONTRIBUTOR_MODEL, "test")

        body = captured_bodies[0]
        # Meta uses the OpenAI Responses shape: input + max_output_tokens +
        # reasoning.effort, NOT messages + max_tokens.
        assert body["max_output_tokens"] == config.FRONTIER_MAX_OUTPUT_TOKENS
        assert body["input"] == "test"
        assert body["temperature"] == pytest.approx(0.2)
        assert body["top_p"] == pytest.approx(1.0)
        assert body["reasoning"] == {"effort": config.META_REASONING_EFFORT}
        assert "max_tokens" not in body
        assert "messages" not in body

    def test_openrouter_cost_usd_correct(self) -> None:
        """Cost = (in/1e6)*in_price + (out/1e6)*out_price using PROVIDER_PRICES."""
        p = _import_providers()
        in_tok, out_tok = 200, 80
        payload = _chat_completions_payload(input_tokens=in_tok, output_tokens=out_tok)

        def mock_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            return payload

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
        ):
            resp = p.OpenRouterClient(transport=mock_transport).generate(
                "qwen/qwen3-7b-fast", "test"
            )

        # No usage.cost in the mock payload → cost falls back to the table.
        in_price, out_price = config.PROVIDER_PRICES["openrouter_default"]
        expected = in_tok / 1e6 * in_price + out_tok / 1e6 * out_price
        assert resp.cost_usd == pytest.approx(expected)

    def test_metaspark_cost_usd_correct(self) -> None:
        p = _import_providers()
        in_tok, out_tok = 150, 60
        payload = _chat_completions_payload(input_tokens=in_tok, output_tokens=out_tok)

        def mock_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            return payload

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"MODEL_API_KEY": "test-key-meta"}),
        ):
            resp = p.MetaSparkClient(transport=mock_transport).generate(
                config.META_CONTRIBUTOR_MODEL, "test"
            )

        in_price, out_price = config.PROVIDER_PRICES["meta_contributor"]
        expected = in_tok / 1e6 * in_price + out_tok / 1e6 * out_price
        assert resp.cost_usd == pytest.approx(expected)

    def test_grok_cost_usd_is_zero(self) -> None:
        """GrokBridgeClient is subscription-backed: cost_usd must always be 0.0."""
        p = _import_providers()

        def mock_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            return _chat_completions_payload(input_tokens=500, output_tokens=160)

        with patch.object(config, "FRONTIER_ENABLED", True):
            resp = p.GrokBridgeClient(transport=mock_transport).generate(
                "grok-1.5-flash", "test"
            )

        assert resp.cost_usd == 0.0, (
            "GrokBridgeClient is subscription-backed; "
            f"cost_usd must be 0.0, got {resp.cost_usd}"
        )

    def test_model_response_is_frozen(self) -> None:
        """ModelResponse must be immutable (frozen dataclass)."""
        p = _import_providers()

        def mock_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            return _chat_completions_payload(input_tokens=10, output_tokens=5)

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
        ):
            resp = p.OpenRouterClient(transport=mock_transport).generate(
                "qwen/qwen3-7b-fast", "test"
            )

        with pytest.raises((AttributeError, TypeError)):
            resp.text = "mutated"  # type: ignore[misc]

    def test_openrouter_posts_to_chat_completions_endpoint(self) -> None:
        """OpenRouterClient must POST to .../chat/completions."""
        p = _import_providers()
        captured_urls: list[str] = []

        def capturing_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            captured_urls.append(url)
            return _chat_completions_payload(input_tokens=10, output_tokens=5)

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
        ):
            p.OpenRouterClient(transport=capturing_transport).generate(
                "qwen/qwen3-7b-fast", "test"
            )

        assert len(captured_urls) == 1
        assert "chat/completions" in captured_urls[0], (
            f"OpenRouterClient should POST to chat/completions, "
            f"got: {captured_urls[0]!r}"
        )

    def test_grok_posts_to_chat_completions_endpoint(self) -> None:
        """GrokBridgeClient must POST to proxy_base/chat/completions."""
        p = _import_providers()
        captured_urls: list[str] = []

        def capturing_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            captured_urls.append(url)
            return _chat_completions_payload(input_tokens=10, output_tokens=5)

        with patch.object(config, "FRONTIER_ENABLED", True):
            p.GrokBridgeClient(transport=capturing_transport).generate(
                "grok-1.5-flash", "test"
            )

        assert len(captured_urls) == 1
        assert "chat/completions" in captured_urls[0]


# ---------------------------------------------------------------------------
# (c) TestRetryOnce
#
# First call raises RuntimeError("HTTP 429 from ...").
# Second call returns a valid 200 payload.
# Transport must be called exactly twice; response is well-formed.
# ---------------------------------------------------------------------------


class TestRetryOnce:
    """LLD-E §5.2: one operational retry on 429/5xx/transport error."""

    def test_429_then_200_succeeds_openrouter(self) -> None:
        p = _import_providers()
        call_count = 0

        def flaky_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError(f"HTTP 429 from {url}: rate limited")
            return _chat_completions_payload(
                text="retry worked", input_tokens=50, output_tokens=20
            )

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
        ):
            resp = p.OpenRouterClient(transport=flaky_transport).generate(
                "qwen/qwen3-7b-fast", "test"
            )

        assert resp.text == "retry worked"
        assert call_count == 2, f"expected exactly 2 transport calls, got {call_count}"

    def test_5xx_then_200_succeeds_metaspark(self) -> None:
        p = _import_providers()
        call_count = 0

        def flaky_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError(f"HTTP 503 from {url}: service unavailable")
            return _chat_completions_payload(
                text="meta retry ok", input_tokens=30, output_tokens=10
            )

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"MODEL_API_KEY": "test-key-meta"}),
        ):
            resp = p.MetaSparkClient(transport=flaky_transport).generate(
                config.META_CONTRIBUTOR_MODEL, "test"
            )

        assert resp.text == "meta retry ok"
        assert call_count == 2

    def test_connection_error_then_200_succeeds(self) -> None:
        """ConnectionError is also retryable (transport failure)."""
        p = _import_providers()
        call_count = 0

        def flaky_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("upstream dropped connection")
            return _chat_completions_payload(
                text="conn retry ok", input_tokens=20, output_tokens=8
            )

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
        ):
            resp = p.OpenRouterClient(transport=flaky_transport).generate(
                "qwen/qwen3-7b-fast", "test"
            )

        assert resp.text == "conn retry ok"
        assert call_count == 2

    def test_grok_429_then_200_succeeds(self) -> None:
        p = _import_providers()
        call_count = 0

        def flaky_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError(f"HTTP 429 from {url}: rate limited")
            return _chat_completions_payload(text="grok retry", input_tokens=40, output_tokens=12)

        with patch.object(config, "FRONTIER_ENABLED", True):
            resp = p.GrokBridgeClient(transport=flaky_transport).generate(
                "grok-1.5-flash", "test"
            )

        assert resp.text == "grok retry"
        assert call_count == 2


# ---------------------------------------------------------------------------
# (d) TestRetryExhausted
#
# Both calls raise a 5xx.  Must raise after two total attempts (no third try).
# The raised exception is the one from the second attempt.
# ---------------------------------------------------------------------------


class TestRetryExhausted:
    """LLD-E §5.2: persistent error → raise after one retry (two total calls)."""

    def test_persistent_5xx_raises_openrouter(self) -> None:
        p = _import_providers()
        call_count = 0

        def always_500(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"HTTP 500 from {url}: internal server error")

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
            pytest.raises(RuntimeError, match="HTTP 500"),
        ):
            p.OpenRouterClient(transport=always_500).generate(
                "qwen/qwen3-7b-fast", "test"
            )

        assert call_count == 2, (
            f"expected exactly 2 calls (1 attempt + 1 retry), got {call_count}"
        )

    def test_persistent_5xx_raises_metaspark(self) -> None:
        p = _import_providers()
        call_count = 0

        def always_502(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"HTTP 502 from {url}: bad gateway")

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"MODEL_API_KEY": "test-key-meta"}),
            pytest.raises(RuntimeError, match="HTTP 502"),
        ):
            p.MetaSparkClient(transport=always_502).generate(
                config.META_CONTRIBUTOR_MODEL, "test"
            )

        assert call_count == 2

    def test_persistent_5xx_raises_grok(self) -> None:
        p = _import_providers()
        call_count = 0

        def always_503(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"HTTP 503 from {url}: service unavailable")

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            pytest.raises(RuntimeError, match="HTTP 503"),
        ):
            p.GrokBridgeClient(transport=always_503).generate("grok-1.5-flash", "test")

        assert call_count == 2

    def test_no_third_attempt(self) -> None:
        """Exactly two transport calls — no infinite retry loop."""
        p = _import_providers()
        call_count = 0

        def always_fail(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"HTTP 500 from {url}: error")

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
            pytest.raises(RuntimeError),
        ):
            p.OpenRouterClient(transport=always_fail).generate(
                "qwen/qwen3-7b-fast", "test"
            )

        assert call_count == 2, (
            f"Retry loop must stop after 2 calls (1 attempt + 1 retry). "
            f"Got {call_count} calls."
        )

    def test_content_failure_not_retried(self) -> None:
        """A KeyError (malformed response) is NOT retryable — must propagate immediately."""
        p = _import_providers()
        call_count = 0

        def bad_payload_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            # Missing 'choices' key — malformed response (content failure).
            return {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
            pytest.raises(KeyError),
        ):
            p.OpenRouterClient(transport=bad_payload_transport).generate(
                "qwen/qwen3-7b-fast", "test"
            )

        # KeyError is not retryable — only one transport call.
        assert call_count == 1, (
            f"Content failures (KeyError) must NOT be retried. Got {call_count} calls."
        )


# ---------------------------------------------------------------------------
# TestResponsesShapeAndCost — Meta Responses parsing, reasoning-model guards,
# and OpenRouter authoritative usage.cost.
# ---------------------------------------------------------------------------


def _responses_payload(
    text: str = "6912",
    input_tokens: int = 27,
    output_tokens: int = 90,
) -> dict:
    """Meta /v1/responses shape: a reasoning item (no content) then a message."""
    return {
        "status": "completed",
        "model": "muse-spark-1.2-contributor",
        "output": [
            {"type": "reasoning", "summary": [], "status": "completed"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": output_tokens - 4},
        },
    }


class TestResponsesShapeAndCost:
    """Responses-API parsing, reasoning-model guards, authoritative cost."""

    def test_metaspark_parses_responses_output_text(self) -> None:
        p = _import_providers()
        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"MODEL_API_KEY": "k"}),
        ):
            resp = p.MetaSparkClient(
                transport=lambda url, body: _responses_payload("6912")  # noqa: ARG005
            ).generate(config.META_CONTRIBUTOR_MODEL, "1234+5678?")
        assert resp.text == "6912"
        assert resp.input_tokens == 27
        assert resp.output_tokens == 90

    def test_metaspark_empty_output_raises(self) -> None:
        """Truncated reasoning model (empty output[]) must raise, not silently pass."""
        p = _import_providers()
        payload = {"status": "incomplete", "output": [], "usage": {"output_tokens": 160}}
        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"MODEL_API_KEY": "k"}),
            pytest.raises(KeyError),
        ):
            p.MetaSparkClient(
                transport=lambda url, body: payload  # noqa: ARG005
            ).generate(config.META_CONTRIBUTOR_MODEL, "x")

    def test_openrouter_prefers_reported_usage_cost(self) -> None:
        """When the payload reports usage.cost (OpenRouter), use it verbatim."""
        p = _import_providers()
        payload = _chat_completions_payload(input_tokens=200, output_tokens=80)
        payload["usage"]["cost"] = 0.00042  # authoritative, != table computation
        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}),
        ):
            resp = p.OpenRouterClient(
                transport=lambda url, body: payload  # noqa: ARG005
            ).generate(config.OPENROUTER_DEFAULT_MODEL, "test")
        assert resp.cost_usd == pytest.approx(0.00042)

    def test_openrouter_null_content_raises_not_none_string(self) -> None:
        """A reasoning model returning content=null must raise, never emit 'None'."""
        p = _import_providers()
        payload = {
            "choices": [
                {
                    "message": {"content": None, "reasoning": "..."},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 160},
            "model": "some/reasoning-model",
        }
        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}),
            pytest.raises(KeyError),
        ):
            p.OpenRouterClient(
                transport=lambda url, body: payload  # noqa: ARG005
            ).generate(config.OPENROUTER_DEFAULT_MODEL, "test")


# ---------------------------------------------------------------------------
# TestMultiRetry (LLD-F §C.2)
#
# _call_with_retries with max_attempts > 2: backs off then succeeds on Nth
# attempt; content error not retried in multi-retry mode.
# ---------------------------------------------------------------------------


class TestMultiRetry:
    """LLD-F §C.2: _call_with_retries with max_attempts=4 (FRONTIER_MAX_RETRIES)."""

    def test_succeeds_on_4th_attempt(self) -> None:
        """Fails 3 times then succeeds on attempt 4 — all 4 calls must be made."""
        p = _import_providers()
        call_count = 0

        def flaky_transport(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise RuntimeError(f"HTTP 429 from {url}: rate limited (call {call_count})")
            return _chat_completions_payload(
                text="succeeded on 4th", input_tokens=50, output_tokens=20
            )

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
        ):
            client = p.OpenRouterClient(transport=flaky_transport)
            resp = client.generate(
                "qwen/qwen3-7b-fast", "test", max_attempts=config.FRONTIER_MAX_RETRIES
            )

        assert resp.text == "succeeded on 4th"
        assert call_count == 4, (
            f"Expected exactly 4 transport calls with max_attempts=4, got {call_count}."
        )

    def test_exhausted_after_max_attempts(self) -> None:
        """Persistent failures raise after exactly max_attempts calls."""
        p = _import_providers()
        call_count = 0

        def always_fail(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"HTTP 503 from {url}: service unavailable")

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
            pytest.raises(RuntimeError, match="HTTP 503"),
        ):
            p.OpenRouterClient(transport=always_fail).generate(
                "qwen/qwen3-7b-fast", "test", max_attempts=config.FRONTIER_MAX_RETRIES
            )

        assert call_count == config.FRONTIER_MAX_RETRIES, (
            f"Expected {config.FRONTIER_MAX_RETRIES} transport calls, got {call_count}."
        )

    def test_content_error_not_retried_in_multi_retry_mode(self) -> None:
        """KeyError (malformed payload) must propagate immediately — not retried."""
        p = _import_providers()
        call_count = 0

        def bad_payload(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            return {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
            pytest.raises(KeyError),
        ):
            p.OpenRouterClient(transport=bad_payload).generate(
                "qwen/qwen3-7b-fast", "test", max_attempts=config.FRONTIER_MAX_RETRIES
            )

        assert call_count == 1, (
            f"Content failures (KeyError) must NOT be retried even with "
            f"max_attempts={config.FRONTIER_MAX_RETRIES}. Got {call_count} calls."
        )

    def test_max_attempts_1_means_no_retry(self) -> None:
        """max_attempts=1: a single failure must raise immediately without retry."""
        p = _import_providers()
        call_count = 0

        def always_fail(url: str, body: dict) -> dict:  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            raise RuntimeError("HTTP 500: error")

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-or"}),
            pytest.raises(RuntimeError),
        ):
            p.OpenRouterClient(transport=always_fail).generate(
                "qwen/qwen3-7b-fast", "test", max_attempts=1
            )

        assert call_count == 1, (
            f"max_attempts=1 must make exactly 1 call. Got {call_count}."
        )
