# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Deterministic task library for the $20-capped empirical harness (LLD-E §3.1).

Each :class:`Task` carries a gold-code ``scorer`` — a pure function that maps
an extracted answer string to ``True`` or ``False`` with NO LLM involvement.
This ensures primary outcomes (Y) are never determined by an opaque judge.

Task families (LLD-E §3.1 items 1–4):
  - Symbolic arithmetic    (arith_*)
  - String manipulation    (str_*)
  - JSON field extraction  (json_*)
  - Simple parsing         (parse_*)

Safety: this module makes zero network or API calls. It does not import from
``agentassert_abc.experiments.config`` because no budget, model, or frontier
constants are needed — tasks are static, offline, and free to evaluate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Soft-scorer default (module-level named function — hashable + debuggable)
# ---------------------------------------------------------------------------


def _soft_nonempty(answer: str) -> bool:
    """Default soft scorer: return True iff *answer* is non-empty.

    Reproduces the pre-LLD-F behavior of ``len(text) > 0`` after extraction.
    Defined at module level (not as a lambda) so it is hashable and has a
    descriptive repr for debugging.

    Args:
        answer: The extracted (lower-cased, stripped) agent response.

    Returns:
        ``True`` when *answer* is non-empty.
    """
    return bool(answer)


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Task:
    """An objectively scorable evaluation task with a deterministic gold judge.

    All fields are immutable. ``scorer`` receives the output of
    :func:`extract_answer` — already stripped and lower-cased — and returns
    ``True`` iff the answer is correct.  ``soft_scorer`` applies a relaxed
    structural check (e.g., all required JSON keys present) that is logged
    separately but does not determine the primary outcome Y.

    Args:
        id: Unique snake_case identifier (e.g. ``"arith_add"``).
        prompt: The question presented to the agent.
        ground_truth: The canonical correct answer as a raw string.
        scorer: Pure function ``(extracted_answer: str) -> bool``.
        domain: Domain label for domain-grounded missions (default ``"generic"``).
        soft_scorer: Relaxed structural checker (default :func:`_soft_nonempty`).
            Must be a pure deterministic function — never an LLM judge.
    """

    id: str
    prompt: str
    ground_truth: str
    scorer: Callable[[str], bool]
    domain: str = "generic"
    soft_scorer: Callable[[str], bool] = field(default=_soft_nonempty)


# ---------------------------------------------------------------------------
# Answer extraction  (deterministic, no LLM)
# ---------------------------------------------------------------------------

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def extract_answer(raw: str) -> str:
    """Normalise agent output into a canonical comparison form.

    Transformations applied (in order):
    1. Strip leading/trailing whitespace.
    2. Lower-case the entire string.
    3. Collapse all internal whitespace runs to a single space.

    Args:
        raw: The raw string emitted by the agent.

    Returns:
        A normalised string suitable for direct comparison with scorers.

    Examples:
        >>> extract_answer("  Hello World  ")
        'hello world'
        >>> extract_answer("6912\\n")
        '6912'
    """
    stripped = raw.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    return _WHITESPACE_RE.sub(" ", lowered)


# ---------------------------------------------------------------------------
# Score dispatch
# ---------------------------------------------------------------------------


def score(task: Task, raw_output: str) -> bool:
    """Extract and score a raw agent output against a task's gold judge.

    Args:
        task: The :class:`Task` whose scorer to invoke.
        raw_output: Unprocessed string from the agent.

    Returns:
        ``True`` if the extracted answer passes the gold scorer.
    """
    return task.scorer(extract_answer(raw_output))


def score_soft(task: Task, raw_output: str) -> bool:
    """Apply the task's soft scorer to a raw agent output.

    Extracts the answer with :func:`extract_answer` (strip, lower-case,
    collapse whitespace) then delegates to ``task.soft_scorer``.  For tasks
    from :data:`TASK_LIBRARY` the default soft scorer is :func:`_soft_nonempty`,
    reproducing the pre-LLD-F ``len(text) > 0`` behavior.

    For domain-grounded tasks (LLD-F §A.2) the soft scorer checks structural
    correctness (e.g., all required JSON keys are present and numeric) without
    verifying arithmetic precision — that is the HARD scorer's job.

    Args:
        task: The :class:`Task` whose soft_scorer to invoke.
        raw_output: Unprocessed string from the agent.

    Returns:
        ``True`` if the extracted answer passes the soft structural check.
    """
    return task.soft_scorer(extract_answer(raw_output))


# ---------------------------------------------------------------------------
# Module-level scorer functions  (named, not lambdas — hashable + debuggable)
# ---------------------------------------------------------------------------


def _score_arith_add(answer: str) -> bool:
    """1234 + 5678 = 6912."""
    return answer == "6912"


def _score_arith_mul(answer: str) -> bool:
    """127 × 43 = 5461."""
    return answer == "5461"


def _score_arith_mod(answer: str) -> bool:
    """1000 mod 7 = 6."""
    return answer == "6"


def _score_arith_frac(answer: str) -> bool:
    """3/4 + 1/8 = 0.875."""
    return answer == "0.875"


def _score_arith_sqrt(answer: str) -> bool:
    """Integer square root of 144 = 12."""
    return answer == "12"


def _score_str_reverse(answer: str) -> bool:
    """reverse('hello') = 'olleh'."""
    return answer == "olleh"


def _score_str_upper(answer: str) -> bool:
    """Count of uppercase letters in 'AgentAssert' = 2 (A at 0, A at 5)."""
    return answer == "2"


def _score_str_len(answer: str) -> bool:
    """len('qualixar') = 8."""
    return answer == "8"


def _score_json_field(answer: str) -> bool:
    """Value of 'status' field = 'active'."""
    return answer == "active"


def _score_json_count(answer: str) -> bool:
    """Number of keys in {"a":1,"b":2,"c":3} = 3."""
    return answer == "3"


def _score_parse_int(answer: str) -> bool:
    """Integer extracted from 'The answer is 42.' = '42'."""
    return answer == "42"


# ---------------------------------------------------------------------------
# Task library
# ---------------------------------------------------------------------------

TASK_LIBRARY: Final[tuple[Task, ...]] = (
    # --- Symbolic arithmetic (LLD-E §3.1 item 3) ---
    Task(
        id="arith_add",
        prompt="What is 1234 + 5678? Reply with the integer only.",
        ground_truth="6912",
        scorer=_score_arith_add,
    ),
    Task(
        id="arith_mul",
        prompt="What is 127 multiplied by 43? Reply with the integer only.",
        ground_truth="5461",
        scorer=_score_arith_mul,
    ),
    Task(
        id="arith_mod",
        prompt=(
            "What is the remainder when 1000 is divided by 7? "
            "Reply with the integer only."
        ),
        ground_truth="6",
        scorer=_score_arith_mod,
    ),
    Task(
        id="arith_frac",
        prompt=(
            "What is the decimal value of 3/4 + 1/8? "
            "Reply with the decimal only (e.g. 0.5)."
        ),
        ground_truth="0.875",
        scorer=_score_arith_frac,
    ),
    Task(
        id="arith_sqrt",
        prompt=(
            "What is the integer square root of 144? Reply with the integer only."
        ),
        ground_truth="12",
        scorer=_score_arith_sqrt,
    ),
    # --- String manipulation (LLD-E §3.1 item 1 / structured extraction) ---
    Task(
        id="str_reverse",
        prompt="Reverse the string 'hello'. Reply with the reversed string only.",
        ground_truth="olleh",
        scorer=_score_str_reverse,
    ),
    Task(
        id="str_upper",
        prompt=(
            "How many uppercase letters are in the string 'AgentAssert'? "
            "Reply with the count only."
        ),
        ground_truth="2",
        scorer=_score_str_upper,
    ),
    Task(
        id="str_len",
        prompt=(
            "How many characters are in the word 'qualixar'? "
            "Reply with the count only."
        ),
        ground_truth="8",
        scorer=_score_str_len,
    ),
    # --- JSON field extraction (LLD-E §3.1 item 1) ---
    Task(
        id="json_field",
        prompt=(
            'Extract the value of the "status" field from this JSON and reply '
            'with it only: {"id": 42, "status": "active", "score": 95}'
        ),
        ground_truth="active",
        scorer=_score_json_field,
    ),
    Task(
        id="json_count",
        prompt=(
            'How many key-value pairs are in this JSON object? '
            'Reply with the integer only: {"a": 1, "b": 2, "c": 3}'
        ),
        ground_truth="3",
        scorer=_score_json_count,
    ),
    # --- Simple parsing (LLD-E §3.1 item 1 / structured extraction) ---
    Task(
        id="parse_int",
        prompt=(
            "Extract the integer from this sentence and reply with it only: "
            "'The answer is 42.'"
        ),
        ground_truth="42",
        scorer=_score_parse_int,
    ),
)
