# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for MCP Server Monitor."""

import pytest

from agentassert_abc.dsl.parser import loads_contract
from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.monitor.mcp_monitor import MCPServerMonitor, ToolCallVerdict

_CONTRACT_YAML = """
contractspec: "0.1"
kind: agent
name: mcp-search-server
description: MCP search tool contract
version: "1.0.0"
governance:
  hard:
    - name: budget-check
      category: governance
      check:
        field: tool.budget_remaining
        gte: 0
preconditions:
  - name: auth-required
    check:
      field: tool.auth_token
      exists: true
invariants:
  hard:
    - name: no-sensitive-data
      check:
        field: tool.sensitive_detected
        equals: false
  soft:
    - name: result-size-limit
      check:
        field: tool.result_count
        lte: 100
      recovery: truncate
recovery:
  strategies:
    - name: truncate
      type: inject_correction
      actions: ["Truncate results to limit"]
"""


class TestMCPServerMonitor:
    """MCP tool call monitoring tests."""

    @pytest.fixture
    def monitor(self) -> MCPServerMonitor:
        return MCPServerMonitor(loads_contract(_CONTRACT_YAML))

    def test_pre_invoke_allowed(self, monitor: MCPServerMonitor) -> None:
        verdict = monitor.check_pre_invoke(
            "search",
            {"auth_token": "valid", "budget_remaining": 10, "query": "hello"},
        )
        assert verdict.allowed is True

    def test_pre_invoke_blocked_by_governance(self) -> None:
        monitor = MCPServerMonitor(
            loads_contract(_CONTRACT_YAML), raise_on_hard=False,
        )
        verdict = monitor.check_pre_invoke(
            "search",
            {"auth_token": "valid", "budget_remaining": -5},
        )
        assert verdict.allowed is False
        assert "budget-check" in verdict.hard_violations

    def test_pre_invoke_blocked_by_precondition(self) -> None:
        monitor = MCPServerMonitor(
            loads_contract(_CONTRACT_YAML), raise_on_hard=False,
        )
        verdict = monitor.check_pre_invoke(
            "search",
            {"budget_remaining": 10},
        )
        assert verdict.allowed is False
        assert "auth-required" in verdict.hard_violations

    def test_pre_invoke_raises_when_enabled(self) -> None:
        monitor = MCPServerMonitor(
            loads_contract(_CONTRACT_YAML), raise_on_hard=True,
        )
        with pytest.raises(ContractBreachError, match="search"):
            monitor.check_pre_invoke("search", {"budget_remaining": -5})

    def test_pre_invoke_no_raise_when_disabled(self) -> None:
        monitor = MCPServerMonitor(
            loads_contract(_CONTRACT_YAML), raise_on_hard=False,
        )
        verdict = monitor.check_pre_invoke(
            "search", {"budget_remaining": -5},
        )
        assert verdict.allowed is False

    def test_post_invoke_all_clear(self, monitor: MCPServerMonitor) -> None:
        verdict = monitor.check_post_invoke(
            "search",
            {"sensitive_detected": False, "result_count": 5},
        )
        assert verdict.allowed is True

    def test_post_invoke_hard_violation(self, monitor: MCPServerMonitor) -> None:
        verdict = monitor.check_post_invoke(
            "search",
            {"sensitive_detected": True, "result_count": 5},
        )
        assert verdict.allowed is False
        assert "no-sensitive-data" in verdict.hard_violations

    def test_post_invoke_soft_violation(self, monitor: MCPServerMonitor) -> None:
        verdict = monitor.check_post_invoke(
            "search",
            {"sensitive_detected": False, "result_count": 150},
        )
        assert verdict.recovery_needed is True
        assert "result-size-limit" in verdict.soft_violations

    def test_tool_state_namespacing(self, monitor: MCPServerMonitor) -> None:
        verdict = monitor.check_pre_invoke(
            "my-tool",
            {"auth_token": "ok", "budget_remaining": 5},
        )
        assert verdict.allowed is True  # fields prefixed with 'tool.'


class TestToolCallVerdict:
    def test_default_allowed(self) -> None:
        v = ToolCallVerdict(allowed=True)
        assert bool(v) is True

    def test_not_allowed(self) -> None:
        v = ToolCallVerdict(allowed=False, hard_violations=["budget"])
        assert bool(v) is False
