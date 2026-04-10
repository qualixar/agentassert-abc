# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""LangGraph adapter — intercept StateGraph node outputs for contract monitoring.

Wraps LangGraph node functions to evaluate their output against a ContractSpec
before the state update is merged. Hard violations raise ContractBreachError,
stopping graph execution immediately.

Patent §3.3: ContractMiddleware wraps any agent graph and intercepts
state transitions.

Usage:
    from agentassert_abc.integrations.langgraph import LangGraphAdapter

    adapter = LangGraphAdapter(contract)

    # Wrap individual nodes (sync or async)
    builder.add_node("agent", adapter.wrap_node(call_model))

    # Or wrap the entire compiled graph
    monitored = adapter.wrap_graph(graph)
    result = monitored.invoke(initial_state)

    # Get session metrics
    summary = adapter.session_summary()

Requires: langgraph (pip install langgraph)
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import threading
from functools import wraps
from typing import TYPE_CHECKING, Any

from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.monitor.session import SessionMonitor

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentassert_abc.models import ContractSpec
    from agentassert_abc.monitor.models import SessionSummary, StepResult

try:
    from langgraph.graph import StateGraph  # noqa: F401
    from langgraph.types import Command  # noqa: F401

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False


def _require_langgraph() -> None:
    if not _HAS_LANGGRAPH:
        msg = (
            "LangGraph is required for this adapter. "
            "Install it with: pip install langgraph"
        )
        raise ImportError(msg)


class LangGraphAdapter:
    """Adapter that monitors LangGraph StateGraph node outputs.

    Intercepts each node's return value (state update dict or Command),
    flattens it, and evaluates against the contract.

    Hard violations raise ContractBreachError, stopping graph execution.
    Soft violations are tracked for session-level metrics.

    Thread-safe: Uses a lock around SessionMonitor.step() calls (C-03).
    """

    def __init__(self, contract: ContractSpec) -> None:
        _require_langgraph()
        self._contract = contract
        self._monitor = SessionMonitor(contract)
        self._lock = threading.Lock()  # C-03: Thread safety

    def wrap_node(
        self, node_fn: Callable[..., Any], *, raise_on_hard: bool = True
    ) -> Callable[..., Any]:
        """Wrap a LangGraph node function with contract monitoring.

        Handles both sync and async node functions (H-09).

        Args:
            node_fn: The original node function (state) -> dict | Command.
            raise_on_hard: If True, raise ContractBreachError on hard violations.

        Returns:
            Wrapped node function with same signature.
        """
        # H-09: Detect async node functions
        if inspect.iscoroutinefunction(node_fn):
            return self._wrap_async_node(node_fn, raise_on_hard=raise_on_hard)
        return self._wrap_sync_node(node_fn, raise_on_hard=raise_on_hard)

    def _wrap_sync_node(
        self, node_fn: Callable[..., Any], *, raise_on_hard: bool
    ) -> Callable[..., Any]:
        @wraps(node_fn)
        def wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            result = node_fn(state, *args, **kwargs)
            self._evaluate_node_output(
                node_fn.__name__, state, result, raise_on_hard
            )
            return result

        return wrapper

    def _wrap_async_node(
        self, node_fn: Callable[..., Any], *, raise_on_hard: bool
    ) -> Callable[..., Any]:
        @wraps(node_fn)
        async def wrapper(
            state: dict[str, Any], *args: Any, **kwargs: Any
        ) -> Any:
            result = await node_fn(state, *args, **kwargs)
            self._evaluate_node_output(
                node_fn.__name__, state, result, raise_on_hard
            )
            return result

        return wrapper

    def _evaluate_node_output(
        self,
        node_name: str,
        pre_state: dict[str, Any],
        result: Any,
        raise_on_hard: bool,
    ) -> None:
        """Evaluate node output against contract (thread-safe)."""
        flat_state = self._flatten_node_output(pre_state, result)
        with self._lock:
            step_result = self._monitor.step(flat_state)

        if raise_on_hard and step_result.hard_violations > 0:
            violated = ", ".join(step_result.violated_names)
            msg = (
                f"Hard contract breach in LangGraph node "
                f"'{node_name}': {step_result.hard_violations} "
                f"violation(s) [{violated}]"
            )
            raise ContractBreachError(msg)

    def wrap_graph(self, compiled_graph: Any) -> _MonitoredGraph:
        """Wrap a compiled LangGraph to monitor all invocations.

        Returns a MonitoredGraph that evaluates state after each invoke().

        Args:
            compiled_graph: A compiled StateGraph (graph.compile()).

        Returns:
            MonitoredGraph with the same invoke interface.
        """
        return _MonitoredGraph(compiled_graph, self._monitor, self._lock)

    def check(self, agent_output: dict[str, Any]) -> StepResult:
        """Manually check a state dict against the contract.

        Args:
            agent_output: Flat dict of state fields.

        Returns:
            StepResult with violation counts and drift score.
        """
        with self._lock:
            return self._monitor.step(agent_output)

    def session_summary(self) -> SessionSummary:
        """Get aggregated session metrics including theta."""
        with self._lock:
            return self._monitor.session_summary()

    @staticmethod
    def _flatten_node_output(
        pre_state: dict[str, Any], node_result: Any
    ) -> dict[str, Any]:
        """Merge node output into pre-state for evaluation.

        Uses deep copy to protect against in-place mutations (M-10).

        Args:
            pre_state: State before node execution.
            node_result: Node return value (dict or Command).

        Returns:
            Merged flat state dict for constraint evaluation.
        """
        merged = copy.deepcopy(pre_state)  # M-10: Deep copy

        if isinstance(node_result, dict):
            merged.update(node_result)
        elif _HAS_LANGGRAPH:
            from langgraph.types import Command

            is_command = isinstance(node_result, Command)
            if is_command and getattr(node_result, "update", None):
                merged.update(node_result.update)

        return merged


class _MonitoredGraph:
    """Wrapper around a compiled LangGraph that monitors invoke results.

    Note: stream() is proxied to the underlying graph WITHOUT monitoring.
    Use wrap_node() for per-node interception during streaming.
    """

    def __init__(
        self,
        graph: Any,
        monitor: SessionMonitor,
        lock: threading.Lock,
    ) -> None:
        self._graph = graph
        self._monitor = monitor
        self._lock = lock

    def invoke(
        self,
        input_state: dict[str, Any],
        *args: Any,
        raise_on_hard: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the graph and evaluate the final state.

        Args:
            input_state: Initial state dict.
            raise_on_hard: Raise ContractBreachError on hard violations.

        Returns:
            Final state dict from graph execution.
        """
        result = self._graph.invoke(input_state, *args, **kwargs)

        if isinstance(result, dict):
            with self._lock:
                step_result = self._monitor.step(result)
            if raise_on_hard and step_result.hard_violations > 0:
                violated = ", ".join(step_result.violated_names)
                msg = (
                    f"Hard contract breach in LangGraph output: "
                    f"{step_result.hard_violations} violation(s) [{violated}]"
                )
                raise ContractBreachError(msg)

        return result

    async def ainvoke(
        self,
        input_state: dict[str, Any],
        *args: Any,
        raise_on_hard: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async invoke the graph and evaluate the final state."""
        if hasattr(self._graph, "ainvoke"):
            result = await self._graph.ainvoke(
                input_state, *args, **kwargs
            )
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: self._graph.invoke(input_state, *args, **kwargs)
            )

        if isinstance(result, dict):
            with self._lock:
                step_result = self._monitor.step(result)
            if raise_on_hard and step_result.hard_violations > 0:
                violated = ", ".join(step_result.violated_names)
                msg = (
                    f"Hard contract breach in LangGraph output: "
                    f"{step_result.hard_violations} violation(s) [{violated}]"
                )
                raise ContractBreachError(msg)

        return result

    def __getattr__(self, name: str) -> Any:
        """Proxy other attributes to the underlying graph.

        M-11: stream() is proxied without monitoring. Use wrap_node()
        for per-node interception during streaming.
        """
        return getattr(self._graph, name)
