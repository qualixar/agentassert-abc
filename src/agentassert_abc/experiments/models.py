# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Model clients for the $20-capped empirical validation harness (LLD-E §4/§6).

Two tiers:
  - :class:`LocalClient` — free Ollama inference; cost_usd is always 0.0.
  - :class:`FrontierClient` — paid frontier API (OpenAI-compatible).
    **Hard-gated** behind :data:`~agentassert_abc.experiments.config.FRONTIER_ENABLED`.
    Raises :class:`FrontierDisabledError` if the flag is False — no network call
    is made in that case.

The injected ``transport`` callable keeps both clients test-friendly: every test
passes a synchronous fake that returns a pre-built dict without touching any
real network.  Production callers omit ``transport`` and get the stdlib-urllib
implementation.

Usage::

    from agentassert_abc.experiments.models import generate

    # Local (free, no gate):
    resp = generate("qwen2.5:7b", "Classify this incident…")
    print(resp.text, resp.cost_usd)  # cost_usd == 0.0

    # Frontier (gated — FRONTIER_ENABLED must be True):
    resp = generate("openai/gpt-5-mini", "Classify this incident…")
    print(resp.text, resp.cost_usd)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from agentassert_abc.exceptions import AgentAssertError

# NOTE: import config as a *module reference*, NOT as individual names.
# Patch-based tests work by mutating module attributes; a local binding
# (e.g. `from config import FRONTIER_ENABLED`) would hold the original
# value and bypass safety-gate patches.
from agentassert_abc.experiments import config

# ---------------------------------------------------------------------------
# Module-local exceptions (LLD-E requirement; both subclass AgentAssertError)
# ---------------------------------------------------------------------------


class FrontierDisabledError(AgentAssertError):
    """Raised when FrontierClient.generate is called with FRONTIER_ENABLED=False.

    No network request is ever made in this case — the gate fires first.
    To enable frontier calls, set ``config.FRONTIER_ENABLED = True`` with
    explicit approval and a sufficient budget.
    """


class FrontierTokenCapError(AgentAssertError):
    """Raised when the API response exceeds the LLD-E §6.2 admission caps.

    Admission caps:
      - input_tokens  ≤ FRONTIER_MAX_INPUT_TOKENS  (800)
      - output_tokens ≤ FRONTIER_MAX_OUTPUT_TOKENS (160)

    If either is violated the response is discarded and this error is raised
    so the budget ledger is not silently over-reported.
    """


# ---------------------------------------------------------------------------
# Frozen response dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Immutable record of a single model call.

    Attributes:
        text:          The model's text output.
        input_tokens:  Prompt token count (0 for local if Ollama omits it).
        output_tokens: Completion token count.
        model:         Model identifier as returned by the provider.
        cost_usd:      Reconstructed dollar cost (0.0 for local calls).
    """

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float


# ---------------------------------------------------------------------------
# Transport type alias and default stdlib implementation
# ---------------------------------------------------------------------------

#: Callable signature for the injected HTTP transport.
#: ``(url, body_dict) -> response_dict``
Transport = Callable[[str, dict], dict]

_DEFAULT_TIMEOUT_SECONDS: Final[int] = 30


def _urllib_transport(url: str, body: dict) -> dict:
    """Stdlib urllib POST — no third-party dependencies required."""
    encoded = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {body_text}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Cannot reach {url}: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# LocalClient — free Ollama inference
# ---------------------------------------------------------------------------


class LocalClient:
    """Calls a local Ollama endpoint.  API cost is always $0.

    Args:
        transport: Optional injectable transport for testing.  Defaults to the
            stdlib-urllib implementation that POSTs to localhost:11434.
    """

    def __init__(self, transport: Transport | None = None) -> None:
        self._transport: Transport = (
            transport if transport is not None else _urllib_transport
        )

    def generate(self, model: str, prompt: str) -> ModelResponse:
        """POST to ``{OLLAMA_URL}/api/generate`` and parse the response.

        Args:
            model:  Ollama model tag, e.g. ``"qwen2.5:7b"``.
            prompt: Raw text prompt.

        Returns:
            :class:`ModelResponse` with ``cost_usd == 0.0``.

        Raises:
            KeyError:         Malformed Ollama response (missing ``"response"`` key).
            ConnectionError:  Transport cannot reach Ollama.
            RuntimeError:     Non-2xx HTTP response from Ollama.
        """
        url = f"{config.OLLAMA_URL}/api/generate"
        body: dict = {"model": model, "prompt": prompt, "stream": False}
        raw = self._transport(url, body)
        return ModelResponse(
            text=raw["response"],  # KeyError if absent — intentionally explicit
            input_tokens=int(raw.get("prompt_eval_count", 0)),
            output_tokens=int(raw.get("eval_count", 0)),
            model=str(raw.get("model", model)),
            cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# FrontierClient — paid API, hard-gated
# ---------------------------------------------------------------------------

_FRONTIER_DEFAULT_URL: Final[str] = "https://openrouter.ai/api/v1/chat/completions"


class FrontierClient:
    """Calls a frontier API endpoint (OpenAI-compatible chat completions).

    **Safety gate**: raises :class:`FrontierDisabledError` before any network
    activity if :data:`~agentassert_abc.experiments.config.FRONTIER_ENABLED`
    is ``False``.

    Token caps and cost are enforced per LLD-E §6.2:
      - input_tokens  ≤ :data:`~config.FRONTIER_MAX_INPUT_TOKENS`
      - output_tokens ≤ :data:`~config.FRONTIER_MAX_OUTPUT_TOKENS`
      - cost_usd = (in/1e6)*MAX_IN_PRICE + (out/1e6)*MAX_OUT_PRICE

    Args:
        transport: Optional injectable transport for testing.
        base_url:  Frontier API endpoint.  Defaults to OpenRouter.
    """

    def __init__(
        self,
        transport: Transport | None = None,
        base_url: str = _FRONTIER_DEFAULT_URL,
    ) -> None:
        self._transport: Transport = (
            transport if transport is not None else _urllib_transport
        )
        self._base_url = base_url

    def generate(self, model: str, prompt: str) -> ModelResponse:
        """Call the frontier API and return a cost-annotated :class:`ModelResponse`.

        Args:
            model:  Provider model identifier, e.g. ``"openai/gpt-5-mini"``.
            prompt: Raw text prompt.

        Returns:
            :class:`ModelResponse` with ``cost_usd > 0`` for non-zero token calls.

        Raises:
            FrontierDisabledError:  ``config.FRONTIER_ENABLED`` is ``False``.
            FrontierTokenCapError:  Provider returned more tokens than the cap.
            KeyError:               Malformed provider response.
            ConnectionError:        Transport error.
        """
        # --- SAFETY GATE (must fire before any I/O) -------------------------
        if not config.FRONTIER_ENABLED:
            raise FrontierDisabledError(
                "Frontier API calls are disabled. "
                "Set config.FRONTIER_ENABLED = True only with explicit approval "
                "and sufficient remaining budget."
            )

        # --- Build request ---------------------------------------------------
        request_body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.FRONTIER_MAX_OUTPUT_TOKENS,
            "temperature": 0.2,
            "top_p": 1.0,
        }
        raw = self._transport(self._base_url, request_body)

        # --- Parse token counts ----------------------------------------------
        usage = raw["usage"]
        input_tokens = int(usage["prompt_tokens"])
        output_tokens = int(usage["completion_tokens"])

        # --- Enforce token caps (LLD-E §6.2) ---------------------------------
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

        # --- Cost reconstruction (LLD-E §6.2 admission-price formula) -------
        cost_usd: float = (
            input_tokens / 1e6 * config.MAX_INPUT_PRICE_PER_M_USD
            + output_tokens / 1e6 * config.MAX_OUTPUT_PRICE_PER_M_USD
        )

        # --- Parse text output -----------------------------------------------
        text = str(raw["choices"][0]["message"]["content"])
        response_model = str(raw.get("model", model))

        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=response_model,
            cost_usd=cost_usd,
        )


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


def generate(
    model: str,
    prompt: str,
    transport: Transport | None = None,
) -> ModelResponse:
    """Route a generation request to the correct client tier.

    Models listed in :data:`~config.LOCAL_MODELS` go to :class:`LocalClient`
    (free, no gate).  All other model identifiers go to :class:`FrontierClient`
    (gated, paid).

    Args:
        model:     Model identifier or Ollama tag.
        prompt:    Raw text prompt.
        transport: Optional injectable transport (forwarded to the client).

    Returns:
        :class:`ModelResponse`.

    Raises:
        FrontierDisabledError: Frontier model requested but gate is closed.
        FrontierTokenCapError: Frontier response exceeded token caps.
    """
    if model in config.LOCAL_MODELS:
        return LocalClient(transport=transport).generate(model=model, prompt=prompt)
    return FrontierClient(transport=transport).generate(model=model, prompt=prompt)
