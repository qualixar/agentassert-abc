# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for package version and basic imports."""


def test_version_exists() -> None:
    from agentassert_abc import __version__

    assert __version__ == "0.1.0"


def test_version_is_string() -> None:
    from agentassert_abc import __version__

    assert isinstance(__version__, str)
