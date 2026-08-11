# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec
`packages/claude-code/tests/test_hook.py` (import paths + fixture path only).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HOOK_PY = (
    Path(__file__).parent.parent.parent / "src" / "agentassert_abc" / "claude_code" / "hook.py"
)
FIXTURES = Path(__file__).parent.parent / "test_gateway" / "fixtures" / "contracts"


def run_hook(stdin_data: dict, contract_path: str | None = None) -> dict:
    env = {}
    if contract_path:
        env["AGENTASSERT_CONTRACT"] = str(contract_path)
    result = subprocess.run(
        [sys.executable, str(HOOK_PY)],
        input=json.dumps(stdin_data),
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, **env},
        check=False,
    )
    return json.loads(result.stdout)


class TestHook:
    def test_allow_tool(self) -> None:
        result = run_hook(
            {"hook_type": "PreToolUse", "tool_name": "Read", "session_id": "s1"},
            contract_path=FIXTURES / "safety-minimal.yaml",
        )
        assert result["action"] == "allow"

    def test_block_tool(self) -> None:
        result = run_hook(
            {"hook_type": "PreToolUse", "tool_name": "bash", "session_id": "s1"},
            contract_path=FIXTURES / "safety-minimal.yaml",
        )
        assert result["action"] == "block"
        assert "violation" in result

    def test_no_contract_env(self) -> None:
        result = run_hook(
            {"hook_type": "PreToolUse", "tool_name": "rm", "session_id": "s1"},
        )
        assert result["action"] == "allow"

    def test_malformed_json(self) -> None:
        result = subprocess.run(
            [sys.executable, str(HOOK_PY)],
            input="not json",
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        data = json.loads(result.stdout)
        assert data["action"] == "allow"

    def test_post_tool_use_allow(self) -> None:
        result = run_hook(
            {"hook_type": "PostToolUse", "tool_name": "Read", "session_id": "s1"},
            contract_path=FIXTURES / "safety-minimal.yaml",
        )
        assert result["action"] == "allow"
