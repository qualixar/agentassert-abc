# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `packages/proxy/tests/test_upstream.py`.

Env var renamed `TYPEC_UPSTREAM_ANTHROPIC` -> `AGENTASSERT_UPSTREAM_ANTHROPIC`
to match `agentassert_abc.proxy.forwarder`'s renamed env-var prefix.
"""

from __future__ import annotations

from agentassert_abc.proxy.forwarder import provider_url


def test_default_anthropic() -> None:
    assert provider_url("anthropic") == "https://api.anthropic.com"


def test_default_openai() -> None:
    assert provider_url("openai") == "https://api.openai.com"


def test_contract_override_anthropic() -> None:
    overrides = {"anthropic": "https://example-provider.test/anthropic"}
    assert provider_url("anthropic", overrides) == "https://example-provider.test/anthropic"


def test_contract_override_openai() -> None:
    overrides = {"openai": "https://example-provider.test/v1"}
    assert provider_url("openai", overrides) == "https://example-provider.test/v1"


def test_contract_override_strips_trailing_slash() -> None:
    overrides = {"anthropic": "https://example-provider.test/anthropic/"}
    assert provider_url("anthropic", overrides) == "https://example-provider.test/anthropic"


def test_env_var_agentassert_upstream_anthropic(monkeypatch) -> None:
    monkeypatch.setenv("AGENTASSERT_UPSTREAM_ANTHROPIC", "https://my-custom.llm/anthropic")
    assert provider_url("anthropic") == "https://my-custom.llm/anthropic"


def test_env_var_anthropic_base_url(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example-provider.test/anthropic")
    assert provider_url("anthropic") == "https://example-provider.test/anthropic"


def test_env_var_openai_base_url(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example-provider.test/v1")
    assert provider_url("openai") == "https://example-provider.test/v1"


def test_contract_overrides_env_var(monkeypatch) -> None:
    """Contract upstream takes priority over env vars."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env-value.example.com")
    overrides = {"anthropic": "https://contract-value.example.com"}
    assert provider_url("anthropic", overrides) == "https://contract-value.example.com"


def test_empty_contract_override_falls_through_to_env(monkeypatch) -> None:
    """Empty string in overrides dict falls through to env var."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env-fallback.example.com")
    overrides = {"anthropic": ""}
    assert provider_url("anthropic", overrides) == "https://env-fallback.example.com"


def test_unknown_provider_returns_empty() -> None:
    assert provider_url("unknown_provider") == ""
