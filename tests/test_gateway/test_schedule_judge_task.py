# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Migrated from agentassert-typec `tests/test_schedule_judge_task.py`.

Covers `_schedule_judge_task`'s sync (`asyncio.run`) and async
(`create_task`) dispatch paths, and its (now-logged, not silent — see
`gateway/enforcer.py`) exception handling.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentassert_abc.gateway.enforcer import _schedule_judge_task
from agentassert_abc.metrics.theta import ThetaScorer
from agentassert_abc.process.models import JudgePredicate


def _make_jp(action: str = "theta_penalty") -> JudgePredicate:
    return JudgePredicate(
        rubric="Is the output helpful?", sample_rate=1.0, model="haiku", action_on_fail=action
    )


def _make_dispatcher(
    result: tuple[bool, float] = (True, 0.001), raises: Exception | None = None
) -> MagicMock:
    dispatcher = MagicMock()
    if raises is not None:
        dispatcher.evaluate = AsyncMock(side_effect=raises)
    else:
        dispatcher.evaluate = AsyncMock(return_value=result)
    dispatcher._model = "haiku"
    return dispatcher


class TestScheduleJudgeTaskSync:
    def test_pass_result_no_penalty(self) -> None:
        theta = ThetaScorer()
        _schedule_judge_task(
            _make_dispatcher((True, 0.001)), _make_jp("theta_penalty"), "output text", "s1", theta
        )
        assert theta.compute() == 1.0

    def test_fail_result_theta_penalty_applied(self) -> None:
        theta = ThetaScorer()
        _schedule_judge_task(
            _make_dispatcher((False, 0.001)), _make_jp("theta_penalty"), "output text", "s1", theta
        )
        assert theta.compute() < 1.0

    def test_fail_result_non_theta_action_no_penalty(self) -> None:
        theta = ThetaScorer()
        _schedule_judge_task(
            _make_dispatcher((False, 0.001)), _make_jp("log"), "output text", "s1", theta
        )
        assert theta.compute() == 1.0

    def test_exception_in_evaluate_logged_not_raised(self) -> None:
        theta = ThetaScorer()
        _schedule_judge_task(
            _make_dispatcher(raises=RuntimeError("network error")),
            _make_jp("theta_penalty"),
            "output text",
            "s1",
            theta,
        )
        assert theta.compute() == 1.0


class TestScheduleJudgeTaskAsync:
    @pytest.mark.asyncio
    async def test_async_pass_result_no_penalty(self) -> None:
        theta = ThetaScorer()
        _schedule_judge_task(
            _make_dispatcher((True, 0.001)), _make_jp("theta_penalty"), "output text", "s1", theta
        )
        await asyncio.sleep(0.05)
        assert theta.compute() == 1.0

    @pytest.mark.asyncio
    async def test_async_fail_result_theta_penalty(self) -> None:
        theta = ThetaScorer()
        _schedule_judge_task(
            _make_dispatcher((False, 0.001)), _make_jp("theta_penalty"), "output text", "s1", theta
        )
        await asyncio.sleep(0.05)
        assert theta.compute() < 1.0

    @pytest.mark.asyncio
    async def test_async_exception_logged_not_raised(self) -> None:
        theta = ThetaScorer()
        _schedule_judge_task(
            _make_dispatcher(raises=ValueError("bad")),
            _make_jp("theta_penalty"),
            "output text",
            "s1",
            theta,
        )
        await asyncio.sleep(0.05)
        assert theta.compute() == 1.0
