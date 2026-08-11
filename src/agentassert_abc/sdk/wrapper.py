# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""wrap() — dispatch to the right provider wrapper by client type.

Ported from agentassert-typec-sdk's `wrapper.py`. `SessionMonitor.from_yaml`
-> `SessionEnforcer.from_yaml` (port-delta §C4).
"""

from __future__ import annotations

from typing import Any

from agentassert_abc.gateway.enforcer import SessionEnforcer


def wrap(client: Any, contract_path: str) -> Any:
    """Wrap an Anthropic or OpenAI client with behavioral contract enforcement.

    Usage::

        from anthropic import Anthropic
        from agentassert_abc.sdk import wrap

        client = wrap(Anthropic(), "contract.yaml")
        # client.messages.create(...) is now enforced

    Supported client types (sync and async, dispatched automatically by
    introspecting the wrapped client's methods):
        - anthropic.Anthropic / anthropic.AsyncAnthropic
        - openai.OpenAI / openai.AsyncOpenAI
    """
    enforcer = SessionEnforcer.from_yaml(contract_path)
    client_type_name = f"{type(client).__module__}.{type(client).__name__}"

    if "anthropic" in client_type_name.lower():
        from agentassert_abc.sdk.wrappers.anthropic_wrapper import WrappedAnthropic

        return WrappedAnthropic(client, enforcer)
    if "openai" in client_type_name.lower():
        from agentassert_abc.sdk.wrappers.openai_wrapper import WrappedOpenAI

        return WrappedOpenAI(client, enforcer)

    raise TypeError(
        f"Unsupported client type: {type(client).__name__}. "
        f"Supported: anthropic.Anthropic, anthropic.AsyncAnthropic, "
        f"openai.OpenAI, openai.AsyncOpenAI. "
        f"For other clients, use the HTTP proxy instead."
    )
