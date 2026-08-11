# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_repetition_guard.py` — 10 tests minimum."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque

import pytest

from agentassert_abc.gateway.compiler import CompiledContract
from agentassert_abc.gateway.content.repetition import evaluate_repetition_guard
from agentassert_abc.gateway.enforcer import SessionEnforcer
from agentassert_abc.gateway.events import PreAction
from agentassert_abc.gateway.persistence import SessionStore
from agentassert_abc.gateway.violation_log import ViolationLog
from agentassert_abc.process.models import (
    ContractSpecExtended,
    InvariantsExtended,
    ProcessInvariants,
    RepetitionGuard,
    TypeCDecision,
)


def _make_compiled(
    window_size: int = 3,
    max_repeats: int = 2,
    action: str = "deny",
    ignore_tools: list[str] | None = None,
) -> CompiledContract:
    guard = RepetitionGuard(
        window_size=window_size,
        max_repeats=max_repeats,
        action=action,
        ignore_tools=ignore_tools or [],
    )
    spec = ContractSpecExtended(
        dsl_version="0.4",
        contractspec="typec/v0.4",
        kind="agent",
        name="test",
        description="test",
        version="0.1.0",
        invariants=InvariantsExtended(process=ProcessInvariants(repetition_guard=guard)),
    )
    return CompiledContract.from_spec(spec)


def _pre(tool: str) -> PreAction:
    return PreAction(session_id="s1", contract_id="test", tool=tool, args={})


def _make_history(*tools: str) -> deque[str]:
    return deque(tools, maxlen=1000)


def _build_hash_counts(history: deque[str], window: int) -> defaultdict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    hist = list(history)
    for i in range(window, len(hist) + 1):
        key = hashlib.md5("|".join(hist[i - window : i]).encode()).hexdigest()  # noqa: S324
        counts[key] += 1
    return counts


def test_no_repetition_passes() -> None:
    compiled = _make_compiled(window_size=3, max_repeats=2)
    history = _make_history("read_file", "bash", "write_file")
    counts = _build_hash_counts(history, 3)
    result = evaluate_repetition_guard(_pre("list_dir"), compiled, history, counts, ViolationLog())
    assert result is None


def test_exact_repeat_triggers() -> None:
    compiled = _make_compiled(window_size=2, max_repeats=2, action="deny")
    history = _make_history("bash", "bash", "bash", "bash")
    counts = _build_hash_counts(history, 2)
    result = evaluate_repetition_guard(_pre("bash"), compiled, history, counts, ViolationLog())
    assert result is not None
    assert result.decision == TypeCDecision.DENY


def test_max_repeats_boundary() -> None:
    compiled = _make_compiled(window_size=2, max_repeats=2, action="deny")
    violations = ViolationLog()

    history = _make_history("x", "y", "x", "y")
    counts = _build_hash_counts(history, 2)
    seq_key_xy = hashlib.md5(b"x|y").hexdigest()  # noqa: S324
    assert counts[seq_key_xy] == 2

    result_y = evaluate_repetition_guard(_pre("y"), compiled, history, counts, violations)
    assert result_y is None

    history2 = _make_history("x", "y", "x", "y", "x")
    counts2 = _build_hash_counts(history2, 2)
    result_deny = evaluate_repetition_guard(_pre("y"), compiled, history2, counts2, violations)
    assert result_deny is not None
    assert result_deny.decision == TypeCDecision.DENY


def test_different_sequences_pass() -> None:
    compiled = _make_compiled(window_size=2, max_repeats=2)
    history = _make_history("read_file", "bash", "write_file", "list_dir")
    counts = _build_hash_counts(history, 2)
    result = evaluate_repetition_guard(
        _pre("read_file"), compiled, history, counts, ViolationLog()
    )
    assert result is None


def test_ignored_tool_not_counted() -> None:
    compiled = _make_compiled(window_size=2, max_repeats=2, ignore_tools=["read_file"])
    history = _make_history("read_file", "read_file", "read_file", "read_file")
    counts = _build_hash_counts(history, 2)
    result = evaluate_repetition_guard(
        _pre("read_file"), compiled, history, counts, ViolationLog()
    )
    assert result is None


def test_window_size_1_not_allowed() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RepetitionGuard(window_size=1, max_repeats=2)


def test_action_warn_not_deny() -> None:
    compiled = _make_compiled(window_size=2, max_repeats=2, action="warn")
    violations = ViolationLog()
    history = _make_history("bash", "bash", "bash", "bash")
    counts = _build_hash_counts(history, 2)
    result = evaluate_repetition_guard(_pre("bash"), compiled, history, counts, violations)
    assert result is None
    v = violations.all_violations()
    assert len(v) == 1
    assert v[0]["kind"] == "soft"


def test_history_not_updated_on_deny() -> None:
    """SessionEnforcer._commit_to_history is only called after an ALLOW result."""
    spec = ContractSpecExtended(
        dsl_version="0.4",
        contractspec="typec/v0.4",
        kind="agent",
        name="test",
        description="test",
        version="0.1.0",
        invariants=InvariantsExtended(
            process=ProcessInvariants(
                repetition_guard=RepetitionGuard(window_size=2, max_repeats=2, action="deny")
            )
        ),
    )
    enforcer = SessionEnforcer(spec)
    enforcer._tool_call_history.extend(["bash", "bash", "bash", "bash"])
    seq_key = hashlib.md5(b"bash|bash").hexdigest()  # noqa: S324
    enforcer._sequence_hash_counts[seq_key] = 3

    history_len_before = len(enforcer._tool_call_history)
    result = enforcer.evaluate(
        PreAction(session_id="s1", contract_id="test", tool="bash", args={})
    )

    assert result.is_deny()
    assert len(enforcer._tool_call_history) == history_len_before
    enforcer.close()


def test_session_reset_clears_repetition() -> None:
    spec = ContractSpecExtended(
        dsl_version="0.4",
        contractspec="typec/v0.4",
        kind="agent",
        name="test",
        description="test",
        version="0.1.0",
        invariants=InvariantsExtended(
            process=ProcessInvariants(
                repetition_guard=RepetitionGuard(window_size=2, max_repeats=2, action="deny")
            )
        ),
    )
    enforcer1 = SessionEnforcer(spec)
    assert len(enforcer1._tool_call_history) == 0
    assert len(enforcer1._sequence_hash_counts) == 0

    enforcer2 = SessionEnforcer(spec)
    assert len(enforcer2._tool_call_history) == 0
    assert len(enforcer2._sequence_hash_counts) == 0
    enforcer1.close()
    enforcer2.close()


def test_repetition_persisted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    db_path = str(tmp_path / "rep-session.db")
    spec = ContractSpecExtended(
        dsl_version="0.4",
        contractspec="typec/v0.4",
        kind="agent",
        name="test",
        description="test",
        version="0.1.0",
        invariants=InvariantsExtended(
            process=ProcessInvariants(
                repetition_guard=RepetitionGuard(window_size=2, max_repeats=2, action="deny")
            )
        ),
    )

    enforcer1 = SessionEnforcer(spec)
    store1 = SessionStore(db_path)
    store1.open()
    enforcer1.attach_store(store1)

    enforcer1._tool_call_history.extend(["bash", "read_file", "bash"])
    enforcer1._sequence_hash_counts["abc123"] = 5
    enforcer1.close()

    enforcer2 = SessionEnforcer(spec)
    store2 = SessionStore(db_path)
    store2.open()
    enforcer2.attach_store(store2)

    assert list(enforcer2._tool_call_history) == ["bash", "read_file", "bash"]
    assert enforcer2._sequence_hash_counts["abc123"] == 5
    enforcer2.close()
