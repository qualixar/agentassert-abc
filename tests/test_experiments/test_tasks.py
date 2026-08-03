# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for agentassert_abc.experiments.tasks — deterministic gold-code scoring.

TDD RED phase: these tests were written BEFORE the implementation.
Gold scorers must be deterministic (no LLM) and objectively checkable.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest

from agentassert_abc.experiments.tasks import (
    TASK_LIBRARY,
    Task,
    extract_answer,
    score,
)


# ---------------------------------------------------------------------------
# Task dataclass contract
# ---------------------------------------------------------------------------


class TestTaskDataclass:
    """Task must be frozen, slotted, and fully typed."""

    def test_task_is_frozen(self) -> None:
        """Task instances must be immutable — normal assignment must raise."""
        task = TASK_LIBRARY[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            task.id = "mutated"  # type: ignore[misc]

    def test_task_has_required_fields(self) -> None:
        """Task must expose id, prompt, ground_truth, and scorer fields."""
        task = TASK_LIBRARY[0]
        assert isinstance(task.id, str)
        assert isinstance(task.prompt, str)
        assert isinstance(task.ground_truth, str)
        assert callable(task.scorer)

    def test_task_scorer_signature(self) -> None:
        """scorer(answer: str) -> bool — accepts a string and returns bool."""
        task = TASK_LIBRARY[0]
        result = task.scorer(task.ground_truth)
        assert isinstance(result, bool)

    def test_task_is_hashable(self) -> None:
        """Frozen dataclass must be hashable (usable in sets/dicts)."""
        task = TASK_LIBRARY[0]
        h = hash(task)
        assert isinstance(h, int)


# ---------------------------------------------------------------------------
# TASK_LIBRARY completeness
# ---------------------------------------------------------------------------


class TestTaskLibrary:
    """TASK_LIBRARY must meet minimum coverage and uniqueness requirements."""

    def test_library_has_at_least_eight_tasks(self) -> None:
        """The spec requires >= 8 objectively scorable tasks."""
        assert len(TASK_LIBRARY) >= 8

    def test_all_ids_are_unique(self) -> None:
        """Every task must have a distinct id."""
        ids = [t.id for t in TASK_LIBRARY]
        assert len(ids) == len(set(ids)), "Duplicate task ids found"

    def test_all_prompts_non_empty(self) -> None:
        """Every task prompt must be a non-empty string."""
        for task in TASK_LIBRARY:
            assert task.prompt.strip(), f"Empty prompt on task {task.id!r}"

    def test_all_ground_truths_non_empty(self) -> None:
        """Every task ground_truth must be a non-empty string."""
        for task in TASK_LIBRARY:
            assert task.ground_truth.strip(), f"Empty ground_truth on task {task.id!r}"

    def test_all_ground_truths_pass_own_scorer(self) -> None:
        """gold_truth must score True under its own scorer (sanity gate)."""
        for task in TASK_LIBRARY:
            assert task.scorer(task.ground_truth), (
                f"Task {task.id!r}: scorer returned False for its own ground_truth"
            )

    def test_empty_string_fails_all_scorers(self) -> None:
        """An empty string must fail every scorer (no vacuous passes)."""
        for task in TASK_LIBRARY:
            assert not task.scorer(""), (
                f"Task {task.id!r}: scorer incorrectly accepted empty string"
            )

    def test_garbage_input_fails_all_scorers(self) -> None:
        """Garbage text must fail every scorer."""
        garbage = "xQzW7!@#NoAnswerHere"
        for task in TASK_LIBRARY:
            assert not task.scorer(garbage), (
                f"Task {task.id!r}: scorer incorrectly accepted garbage input"
            )

    def test_covers_arithmetic_domain(self) -> None:
        """At least one arithmetic task must exist (LLD-E §3.1 item 3)."""
        arith_ids = [t.id for t in TASK_LIBRARY if "arith" in t.id]
        assert len(arith_ids) >= 1, "No arithmetic task found in TASK_LIBRARY"

    def test_covers_string_domain(self) -> None:
        """At least one string manipulation task must exist (LLD-E §3.1)."""
        str_ids = [t.id for t in TASK_LIBRARY if "str" in t.id]
        assert len(str_ids) >= 1, "No string task found in TASK_LIBRARY"

    def test_covers_json_domain(self) -> None:
        """At least one JSON extraction task must exist (LLD-E §3.1 item 1)."""
        json_ids = [t.id for t in TASK_LIBRARY if "json" in t.id]
        assert len(json_ids) >= 1, "No JSON extraction task found in TASK_LIBRARY"


# ---------------------------------------------------------------------------
# extract_answer — deterministic normalisation
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    """extract_answer(raw) must strip, case-normalise, and be idempotent."""

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert extract_answer("  hello  ") == "hello"

    def test_lowercases_output(self) -> None:
        assert extract_answer("HELLO") == "hello"

    def test_collapses_internal_whitespace(self) -> None:
        assert extract_answer("  hello   world  ") == "hello world"

    def test_idempotent(self) -> None:
        raw = "  Some Answer  "
        assert extract_answer(extract_answer(raw)) == extract_answer(raw)

    def test_empty_string_returns_empty(self) -> None:
        assert extract_answer("") == ""

    def test_newlines_collapsed(self) -> None:
        assert extract_answer("hello\nworld") == "hello world"


# ---------------------------------------------------------------------------
# score() — end-to-end dispatch
# ---------------------------------------------------------------------------


class TestScoreFunction:
    """score(task, raw_output) -> bool — orchestrates extract + scorer."""

    def test_score_correct_answer_returns_true(self) -> None:
        """A correctly formatted answer must return True."""
        for task in TASK_LIBRARY:
            assert score(task, task.ground_truth), (
                f"Task {task.id!r}: score() returned False for ground_truth"
            )

    def test_score_wrong_answer_returns_false(self) -> None:
        """A clearly wrong answer must return False for every task."""
        for task in TASK_LIBRARY:
            assert not score(task, "wrong_answer_xyz_999"), (
                f"Task {task.id!r}: score() incorrectly returned True for wrong answer"
            )

    def test_score_tolerates_extra_whitespace(self) -> None:
        """Ground truth wrapped in whitespace must still score True."""
        for task in TASK_LIBRARY:
            padded = f"  {task.ground_truth}  "
            assert score(task, padded), (
                f"Task {task.id!r}: score() failed on whitespace-padded ground_truth"
            )

    def test_score_tolerates_uppercase_output(self) -> None:
        """If the correct answer is alphanumeric, uppercase form must still pass."""
        for task in TASK_LIBRARY:
            upper = task.ground_truth.upper()
            assert score(task, upper), (
                f"Task {task.id!r}: score() failed on uppercased ground_truth"
            )


# ---------------------------------------------------------------------------
# Per-task scorer correctness — wrong-answer false negatives
# ---------------------------------------------------------------------------


class TestPerTaskScorers:
    """Each task: correct answer → True, nearby wrong answers → False."""

    def _get_task(self, task_id: str) -> Task:
        for t in TASK_LIBRARY:
            if t.id == task_id:
                return t
        pytest.skip(f"Task {task_id!r} not in TASK_LIBRARY")

    # --- Arithmetic ---

    def test_arith_add_correct(self) -> None:
        t = self._get_task("arith_add")
        assert score(t, t.ground_truth)

    def test_arith_add_off_by_one(self) -> None:
        t = self._get_task("arith_add")
        # Ground truth for 1234+5678 is "6912" — off-by-one must fail
        wrong = str(int(t.ground_truth) + 1)
        assert not score(t, wrong)

    def test_arith_mul_correct(self) -> None:
        t = self._get_task("arith_mul")
        assert score(t, t.ground_truth)

    def test_arith_mul_wrong(self) -> None:
        t = self._get_task("arith_mul")
        assert not score(t, "9999")

    def test_arith_mod_correct(self) -> None:
        t = self._get_task("arith_mod")
        assert score(t, t.ground_truth)

    def test_arith_mod_wrong(self) -> None:
        t = self._get_task("arith_mod")
        assert not score(t, "5")

    # --- String manipulation ---

    def test_str_reverse_correct(self) -> None:
        t = self._get_task("str_reverse")
        assert score(t, t.ground_truth)

    def test_str_reverse_wrong(self) -> None:
        t = self._get_task("str_reverse")
        assert not score(t, "hello")  # original, not reversed

    def test_str_upper_correct(self) -> None:
        # str_upper asks for a count of uppercase letters — answer is numeric
        t = self._get_task("str_upper")
        assert score(t, t.ground_truth)

    def test_str_upper_wrong(self) -> None:
        t = self._get_task("str_upper")
        # off-by-one count must fail
        assert not score(t, "1")

    def test_str_len_correct(self) -> None:
        t = self._get_task("str_len")
        assert score(t, t.ground_truth)

    def test_str_len_wrong(self) -> None:
        t = self._get_task("str_len")
        assert not score(t, "7")

    # --- JSON extraction ---

    def test_json_field_correct(self) -> None:
        t = self._get_task("json_field")
        assert score(t, t.ground_truth)

    def test_json_field_wrong(self) -> None:
        t = self._get_task("json_field")
        assert not score(t, "inactive")

    def test_json_count_correct(self) -> None:
        t = self._get_task("json_count")
        assert score(t, t.ground_truth)

    def test_json_count_wrong(self) -> None:
        t = self._get_task("json_count")
        assert not score(t, "4")

    # --- Parsing ---

    def test_parse_int_correct(self) -> None:
        t = self._get_task("parse_int")
        assert score(t, t.ground_truth)

    def test_parse_int_wrong(self) -> None:
        t = self._get_task("parse_int")
        assert not score(t, "43")
