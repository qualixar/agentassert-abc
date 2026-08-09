# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Frontier provider adapters for the $20-capped validation experiment — Task #20.

Implements the :class:`~agentassert_abc.experiments.motifs.ModelClient` protocol
for three frontier providers.  All adapters share a common OpenAI chat/completions
base and are permanently inert until an authorized operator sets
:data:`~agentassert_abc.experiments.config.FRONTIER_ENABLED` to ``True``.

Adapters
--------
:class:`OpenRouterClient`
    OpenRouter API — ``https://openrouter.ai/api/v1/chat/completions``.
    API key: env var ``OPENROUTER_API_KEY``.

:class:`MetaSparkClient`
    Meta Contributor API — ``https://api.meta.ai/v1/responses``.
    Assumes OpenAI chat/completions response format (OpenAI-compatible endpoint).
    API key: env var ``MODEL_API_KEY``.

:class:`GrokBridgeClient`
    Local hermes proxy (no API key — subscription-backed).
    Endpoint: ``{GROK_PROXY_BASE_URL}/chat/completions``.
    Base URL: env var ``GROK_PROXY_BASE_URL`` (default
    :data:`~agentassert_abc.experiments.config.GROK_PROXY_BASE_URL`).

SAFETY INVARIANTS (enforced; never relaxed)
-------------------------------------------
1. All three adapters raise :class:`FrontierDisabledError` from ``__init__``
   when ``config.FRONTIER_ENABLED`` is ``False``.  No network call is made.
2. ``generate()`` re-checks the gate (belt-and-suspenders) in case the flag is
   patched after construction.
3. Sampling is frozen: ``temperature=0.2``, ``top_p=1.0``,
   ``max_tokens=FRONTIER_MAX_OUTPUT_TOKENS`` (160).
4. Prompts are truncated client-side if they exceed
   ``FRONTIER_MAX_INPUT_TOKENS * _CHARS_PER_TOKEN`` characters.
5. Exactly one retry on 429 / 5xx / transport error (LLD-E §5.2).
   Content failures (e.g. ``KeyError`` from a malformed response) are **not**
   retried.
6. ``cost_usd`` is computed from :data:`~config.PROVIDER_PRICES`.
7. The injected ``transport`` callable keeps all adapters testable without any
   real network socket.

§6.3 batch-gate integration note
---------------------------------
When these adapters are used with ``_execute_mission_batch`` (run.py), pass::

    per_call_ceiling=config.PER_CALL_CEILING_USD

to arm the §6.3 prospective budget gate.  This is NOT done automatically;
the caller must supply it.  Since ``FRONTIER_ENABLED`` is ``False`` by default,
frontier adapters are structurally inert until explicitly enabled.

References
----------
LLD-E-experiment-design-v2.md §4.1 (roster/conditions), §5.1 (sampling),
§5.2 (one retry), §6.2 (admission ceilings), §7.2 (cost/token fields).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Final

# NOTE: import config as a *module reference* so that patch.object works in
# tests — a `from config import FRONTIER_ENABLED` binding would capture the
# original value and bypass safety-gate patches.
from agentassert_abc.experiments import config
from agentassert_abc.experiments.models import (
    FrontierDisabledError,
    FrontierTokenCapError,
    ModelResponse,
)

__all__ = [
    "FrontierDisabledError",
    "GrokBridgeClient",
    "MetaSparkClient",
    "OpenRouterClient",
]

# Re-export so callers only need one import.
FrontierDisabledError = FrontierDisabledError  # noqa: PLW0127 (intentional re-export)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Conservative characters-per-token ratio for client-side prompt truncation.
#: 4 chars/token is a widely used heuristic for English text.  Truncating at
#: FRONTIER_MAX_INPUT_TOKENS * 4 = 3 200 chars prevents obviously oversized
#: prompts from reaching the provider; token cap enforcement is authoritative
#: post-response (LLD-E §6.2).
_CHARS_PER_TOKEN: Final[int] = 4

_DEFAULT_TIMEOUT_SECONDS: Final[int] = 30

# Matches "HTTP 429 ..." and "HTTP 5xx ..." in RuntimeError messages emitted
# by the stdlib urllib transport.
_RETRYABLE_RE: Final[re.Pattern[str]] = re.compile(r"\bHTTP (429|5\d{2})\b")

# Hardcoded provider endpoint roots — the only non-localhost URL in this module.
# These are never contacted unless FRONTIER_ENABLED is explicitly True and the
# adapter is called with a real (non-mock) transport.
_OPENROUTER_DEFAULT_BASE: Final[str] = "https://openrouter.ai/api/v1"
_META_SPARK_ENDPOINT: Final[str] = "https://api.meta.ai/v1/responses"

# ---------------------------------------------------------------------------
# Transport type alias (same as models.py for interoperability)
# ---------------------------------------------------------------------------

#: ``(url, body_dict) -> response_dict`` — injectable HTTP transport.
Transport = Callable[[str, dict], dict]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_retryable(exc: Exception) -> bool:
    """Return ``True`` for errors that warrant exactly one retry (LLD-E §5.2).

    Retryable:
    - :exc:`ConnectionError` — transport / network failure.
    - :exc:`RuntimeError` whose message matches ``HTTP 429`` or ``HTTP 5xx``
      (rate-limit or transient server error).

    Not retryable (must propagate immediately):
    - :exc:`KeyError` / :exc:`ValueError` — malformed response (content failure).
    - Any other exception type.
    """
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, RuntimeError):
        return bool(_RETRYABLE_RE.search(str(exc)))
    return False


def _make_http_transport(
    extra_headers: dict[str, str] | None = None,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
) -> Transport:
    """Return a stdlib-urllib ``Transport`` that merges *extra_headers* into each POST.

    This factory is called at adapter construction time when no mock transport is
    injected.  The returned closure captures *extra_headers* (including any Bearer
    token) and performs real HTTPS POSTs — it must **never** be called unless
    ``config.FRONTIER_ENABLED`` is ``True`` and the adapter's gate has passed.

    Args:
        extra_headers: Additional HTTP headers (e.g.
            ``{"Authorization": "Bearer sk-…"}``).  Merged with
            ``Content-Type: application/json``.
        timeout: Socket timeout in seconds.

    Returns:
        A ``Transport`` callable ``(url, body) -> response_dict``.

    Raises:
        RuntimeError:     HTTP error response (status code included in message).
        ConnectionError:  URL/network error.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    def _transport(url: str, body: dict) -> dict:
        encoded = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=encoded,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {exc.code} from {url}: {body_text}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Cannot reach {url}: {exc.reason}") from exc

    return _transport


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------


class _OpenAICompatBase:
    """Shared logic for all three OpenAI-compatible frontier adapters.

    Subclasses call :meth:`__init__` with provider-specific parameters and
    inherit ``generate()``, ``_call_with_one_retry()``, and
    ``_parse_response()``.

    The gate is checked twice: once in ``__init__`` (fail fast at construction)
    and once in ``generate()`` (belt-and-suspenders in case the flag is patched
    between construction and the first call).
    """

    __slots__ = ("_endpoint", "_price_key", "_transport")

    def __init__(
        self,
        *,
        endpoint: str,
        env_key_name: str | None,
        price_key: str,
        transport: Transport | None,
    ) -> None:
        """Validate the frontier gate and initialise the adapter.

        Args:
            endpoint:     Full URL to POST to (e.g. the chat/completions endpoint).
            env_key_name: Environment variable name for the Bearer API key, or
                          ``None`` if the provider needs no key (GrokBridgeClient).
            price_key:    Key into :data:`~config.PROVIDER_PRICES` for cost
                          computation.
            transport:    Optional injectable transport for testing.  When
                          ``None``, a real urllib transport is built using the
                          API key from the environment.

        Raises:
            FrontierDisabledError: ``config.FRONTIER_ENABLED`` is ``False``, OR
                the required API key env var is absent.
        """
        # --- GATE CHECK #1: fail fast at construction -----------------------
        if not config.FRONTIER_ENABLED:
            raise FrontierDisabledError(
                "Frontier API calls are disabled. "
                "Set config.FRONTIER_ENABLED = True only with explicit approval "
                "and sufficient remaining budget. Current value: False."
            )

        # --- Credential check -----------------------------------------------
        api_key: str | None = None
        if env_key_name is not None:
            api_key = os.environ.get(env_key_name)
            if not api_key:
                raise FrontierDisabledError(
                    f"Frontier API key missing: environment variable "
                    f"{env_key_name!r} is not set. "
                    "Set the key AND config.FRONTIER_ENABLED = True to enable "
                    "frontier calls."
                )

        self._endpoint: str = endpoint
        self._price_key: str = price_key

        # --- Transport selection --------------------------------------------
        # Tests inject a mock transport (in-process, zero sockets).
        # Production code gets a real urllib transport with Bearer auth.
        if transport is not None:
            self._transport: Transport = transport
        elif api_key:
            self._transport = _make_http_transport(
                {"Authorization": f"Bearer {api_key}"}
            )
        else:
            # GrokBridgeClient — subscription-backed local proxy, no auth header.
            self._transport = _make_http_transport()

    def generate(self, model: str, prompt: str) -> ModelResponse:
        """Call the provider and return a cost-annotated :class:`~.models.ModelResponse`.

        Args:
            model:  Provider model identifier.
            prompt: Raw text prompt.

        Returns:
            :class:`~.models.ModelResponse` with ``cost_usd`` from
            :data:`~config.PROVIDER_PRICES`.

        Raises:
            FrontierDisabledError:   Gate is closed (``FRONTIER_ENABLED=False``).
            FrontierTokenCapError:   Response exceeded LLD-E §6.2 token caps.
            KeyError:                Malformed provider response.
            ConnectionError:         Transport failure (after one retry).
            RuntimeError:            HTTP 4xx/5xx from provider (after one retry
                                     for 429/5xx; immediately for other codes).
        """
        # --- GATE CHECK #2: belt-and-suspenders ----------------------------
        if not config.FRONTIER_ENABLED:
            raise FrontierDisabledError(
                "Frontier API calls are disabled (FRONTIER_ENABLED is False). "
                "No network request has been made."
            )

        # --- Client-side prompt truncation (conservative heuristic) --------
        max_chars: int = config.FRONTIER_MAX_INPUT_TOKENS * _CHARS_PER_TOKEN
        safe_prompt: str = prompt[:max_chars]

        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": safe_prompt}],
            "max_tokens": config.FRONTIER_MAX_OUTPUT_TOKENS,  # LLD-E §5.1
            "temperature": 0.2,                               # LLD-E §5.1
            "top_p": 1.0,                                     # LLD-E §5.1
        }

        raw = self._call_with_one_retry(body)
        return self._parse_response(raw, model)

    def _call_with_one_retry(self, body: dict) -> dict:
        """Execute the transport, retrying once on 429/5xx/transport error.

        LLD-E §5.2: one operational retry on transport errors, rate limits, or
        provider 5xx responses.  Content failures are never retried.

        Args:
            body: Request body dict.

        Returns:
            Parsed response dict from the provider.

        Raises:
            Any exception from the second attempt (non-retryable first
            exceptions re-raise immediately).
        """
        try:
            return self._transport(self._endpoint, body)
        except Exception as first_exc:
            if not _is_retryable(first_exc):
                raise
            # One retry — the second exception propagates as-is.
            return self._transport(self._endpoint, body)

    def _parse_response(self, raw: dict, model: str) -> ModelResponse:
        """Parse an OpenAI chat/completions response and return a ModelResponse.

        Handles both ``prompt_tokens``/``completion_tokens`` (standard chat
        completions) and ``input_tokens``/``output_tokens`` (OpenAI Responses
        API variant) field names.

        Args:
            raw:   Parsed JSON response dict from the provider.
            model: Model identifier (used as fallback if ``"model"`` absent
                   from response).

        Returns:
            :class:`~.models.ModelResponse` with tokens, text, and cost.

        Raises:
            FrontierTokenCapError: Either token count exceeds the LLD-E §6.2 cap.
            KeyError:              Required field missing from response.
        """
        usage: dict = raw["usage"]

        # Support both OpenAI chat/completions and Responses API field names.
        if "prompt_tokens" in usage:
            input_tokens = int(usage["prompt_tokens"])
        elif "input_tokens" in usage:
            input_tokens = int(usage["input_tokens"])
        else:
            input_tokens = 0

        if "completion_tokens" in usage:
            output_tokens = int(usage["completion_tokens"])
        elif "output_tokens" in usage:
            output_tokens = int(usage["output_tokens"])
        else:
            output_tokens = 0

        # --- Token cap enforcement (LLD-E §6.2) ----------------------------
        if input_tokens > config.FRONTIER_MAX_INPUT_TOKENS:
            raise FrontierTokenCapError(
                f"Input tokens {input_tokens} exceed cap "
                f"{config.FRONTIER_MAX_INPUT_TOKENS} (LLD-E §6.2). "
                "Response discarded."
            )
        if output_tokens > config.FRONTIER_MAX_OUTPUT_TOKENS:
            raise FrontierTokenCapError(
                f"Output tokens {output_tokens} exceed cap "
                f"{config.FRONTIER_MAX_OUTPUT_TOKENS} (LLD-E §6.2). "
                "Response discarded."
            )

        # --- Cost computation from PROVIDER_PRICES -------------------------
        fallback_prices = (
            config.MAX_INPUT_PRICE_PER_M_USD,
            config.MAX_OUTPUT_PRICE_PER_M_USD,
        )
        in_price, out_price = config.PROVIDER_PRICES.get(
            self._price_key, fallback_prices
        )
        cost_usd: float = (
            input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price
        )

        # --- Parse text (OpenAI chat/completions format) -------------------
        text: str = str(raw["choices"][0]["message"]["content"])
        response_model: str = str(raw.get("model", model))

        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=response_model,
            cost_usd=cost_usd,
        )


# ---------------------------------------------------------------------------
# OpenRouterClient
# ---------------------------------------------------------------------------


class OpenRouterClient(_OpenAICompatBase):
    """OpenRouter frontier adapter — ``https://openrouter.ai/api/v1/chat/completions``.

    Satisfies the :class:`~agentassert_abc.experiments.motifs.ModelClient`
    protocol via duck typing.

    **API key:** environment variable ``OPENROUTER_API_KEY``.

    **Cost:** uses ``PROVIDER_PRICES["openrouter_qwen_flash"]``
    ($0.03 / $0.13 per 1M tokens).

    Args:
        transport: Optional injectable transport for testing.  Defaults to a
            real urllib HTTPS transport with Bearer auth.
        base_url:  Override the OpenRouter base URL.  Defaults to
            ``https://openrouter.ai/api/v1``.

    Raises:
        FrontierDisabledError: ``config.FRONTIER_ENABLED`` is ``False`` OR
            ``OPENROUTER_API_KEY`` is not set.

    Example (requires ``FRONTIER_ENABLED = True`` and key set):
        >>> client = OpenRouterClient()
        >>> resp = client.generate("qwen/qwen3-7b-fast", "What is 2+2?")
        >>> print(resp.text, resp.cost_usd)
    """

    def __init__(
        self,
        transport: Transport | None = None,
        base_url: str | None = None,
    ) -> None:
        base = base_url or _OPENROUTER_DEFAULT_BASE
        super().__init__(
            endpoint=f"{base}/chat/completions",
            env_key_name="OPENROUTER_API_KEY",
            price_key="openrouter_qwen_flash",
            transport=transport,
        )


# ---------------------------------------------------------------------------
# MetaSparkClient
# ---------------------------------------------------------------------------


class MetaSparkClient(_OpenAICompatBase):
    """Meta Contributor API adapter — ``https://api.meta.ai/v1/responses``.

    Posts directly to the Meta Responses endpoint (the full URL is the
    endpoint; no ``/chat/completions`` suffix is appended).  The response is
    parsed as OpenAI chat/completions format (Meta's OpenAI-compatible
    interface).

    **API key:** environment variable ``MODEL_API_KEY``.

    **Default model:** :data:`~config.META_CONTRIBUTOR_MODEL`.

    **Cost:** uses ``PROVIDER_PRICES["meta_contributor"]``
    ($0.10 / $0.20 per 1M tokens).

    Args:
        transport: Optional injectable transport for testing.
        base_url:  Override the Meta endpoint URL.  Defaults to
            ``https://api.meta.ai/v1/responses``.

    Raises:
        FrontierDisabledError: ``config.FRONTIER_ENABLED`` is ``False`` OR
            ``MODEL_API_KEY`` is not set.
    """

    def __init__(
        self,
        transport: Transport | None = None,
        base_url: str | None = None,
    ) -> None:
        # The Meta endpoint IS the full URL — no /chat/completions suffix.
        endpoint = base_url or _META_SPARK_ENDPOINT
        super().__init__(
            endpoint=endpoint,
            env_key_name="MODEL_API_KEY",
            price_key="meta_contributor",
            transport=transport,
        )


# ---------------------------------------------------------------------------
# GrokBridgeClient
# ---------------------------------------------------------------------------


class GrokBridgeClient(_OpenAICompatBase):
    """Local hermes proxy adapter for Grok — no API key required.

    Routes requests through a locally running proxy (default
    ``http://localhost:8787/v1/chat/completions``).  The proxy is
    subscription-backed; per-call cost is always $0.00.

    The base URL is read from env var ``GROK_PROXY_BASE_URL`` at construction
    time; falls back to :data:`~config.GROK_PROXY_BASE_URL`
    (``"http://localhost:8787/v1"``).

    **Gate:** still requires ``config.FRONTIER_ENABLED = True`` — the gate
    applies to all frontier adapters regardless of key requirement.

    **Cost:** uses ``PROVIDER_PRICES["grok_bridge"]`` (0.00 / 0.00 per 1M
    tokens — subscription-backed flat cost).

    Args:
        transport: Optional injectable transport for testing.

    Raises:
        FrontierDisabledError: ``config.FRONTIER_ENABLED`` is ``False``.
    """

    def __init__(self, transport: Transport | None = None) -> None:
        base = os.environ.get("GROK_PROXY_BASE_URL", config.GROK_PROXY_BASE_URL)
        super().__init__(
            endpoint=f"{base}/chat/completions",
            env_key_name=None,  # subscription-backed; no API key required
            price_key="grok_bridge",
            transport=transport,
        )
