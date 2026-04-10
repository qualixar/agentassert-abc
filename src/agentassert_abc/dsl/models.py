# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""DSL-specific models — ParseResult and ValidationError."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from agentassert_abc.models import ContractSpec  # noqa: TCH001


class ValidationError(BaseModel):
    """A single validation finding."""

    model_config = ConfigDict(frozen=True)

    level: Literal["error", "warning"]
    path: str
    message: str
    code: str


class ParseResult(BaseModel):
    """Result of parsing a contract YAML."""

    model_config = ConfigDict(frozen=True)

    contract: ContractSpec | None = None
    errors: list[ValidationError] = []

    @property
    def is_valid(self) -> bool:
        """True if contract parsed with no errors."""
        return self.contract is not None and all(e.level != "error" for e in self.errors)
