# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""SDK — `wrap(client, contract_path)` drop-in contract enforcement.

Ported from `agentassert-typec-sdk` (MIT) into `agentassert_abc.sdk`
(AGPL-3.0-or-later).

Usage::

    from anthropic import Anthropic
    from agentassert_abc.sdk import wrap

    client = wrap(Anthropic(), "contract.yaml")
    client.messages.create(...)  # now enforced

Install with the ``sdk`` extra: ``pip install agentassert-abc[sdk]``.
"""

from __future__ import annotations

from agentassert_abc.sdk.wrapper import wrap

__all__ = ["wrap"]
