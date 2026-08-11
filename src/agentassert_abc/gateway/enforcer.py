# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""SessionEnforcer — event-driven real-time enforcement orchestrator.

Ported from agentassert-typec's `monitor/session.py::SessionMonitor`
(port-delta §C4, item #7 — FATAL rename). abc v2 already has a
`SessionMonitor` (the *measurement* plane, step-driven,
`agentassert_abc.monitor.session.SessionMonitor`). Reusing that name here
would silently shadow one or the other on import. This class is the typec
enforcement plane, renamed `SessionEnforcer`.

Wiring changes from typec (see `gateway/engine.py` module docstring for the
full compliance/drift/theta rationale):
- `self._drift` is `agentassert_abc.metrics.drift.DriftTracker` (abc v2's),
  NOT typec's discarded tracker.
- `self._theta` is `agentassert_abc.metrics.theta.ThetaScorer` (Phase B),
  which uses the §5.7 `(c_hard + c_soft) / 2` formula, NOT typec's
  `0.7*c_hard + 0.3*c_soft`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from collections import Counter, defaultdict, deque
from typing import TYPE_CHECKING, Any

from agentassert_abc.exceptions import ContractLoadError
from agentassert_abc.gateway.compiler import CompiledContract
from agentassert_abc.gateway.engine import dispatch_event
from agentassert_abc.gateway.events import DriftReport, PreAction, SessionEnd, TurnEnd, TypeCEvent
from agentassert_abc.gateway.judge import JudgeDispatcher
from agentassert_abc.gateway.violation_log import ViolationLog
from agentassert_abc.metrics.drift import DriftTracker
from agentassert_abc.metrics.theta import ThetaScorer

if TYPE_CHECKING:
    from agentassert_abc.gateway.persistence import SessionStore
    from agentassert_abc.process.models import ContractSpecExtended, DecisionResult

logger = logging.getLogger(__name__)

#: Θ penalty applied when a sampled judge_predicate FAILs (action_on_fail
#: == "theta_penalty"). Distinct from process_drift's penalty (0.05).
_JUDGE_FAIL_PENALTY = 0.03


class SessionEnforcer:
    """Event-dispatch orchestrator for a single agent session.

    Usage::

        enforcer = SessionEnforcer.from_yaml("contract.yaml")
        decision = enforcer.evaluate(PreAction(session_id=..., contract_id=..., tool="search"))
        if decision.is_deny():
            ...
        end = enforcer.close()
    """

    def __init__(self, contract: ContractSpecExtended) -> None:
        self._contract = contract
        self._compiled: CompiledContract = CompiledContract.from_spec(contract)
        self._drift_window = contract.drift.window if contract.drift else 50
        self._drift = DriftTracker(config=contract.drift if contract.drift else None)
        self._theta = ThetaScorer(
            weights=contract.reliability.weights if contract.reliability else None
        )
        self._violations = ViolationLog()
        self._lock = threading.RLock()
        self._turn_count = 0
        self._deny_count = 0
        self._judge_dispatchers: list[JudgeDispatcher] = []
        self._seen_tools_session: set[str] = set()
        self._seen_tools_turn: set[str] = set()

        # Persistence store — attached externally via attach_store().
        self._store: SessionStore | None = None

        # Phase 3: cost tracking.
        self._accumulated_cost_usd: float = 0.0
        self._cost_lock = threading.Lock()

        # Phase 3: repetition guard — MUST have maxlen=1000.
        self._tool_call_history: deque[str] = deque(maxlen=1000)
        self._sequence_hash_counts: defaultdict[str, int] = defaultdict(int)

        self._init_judge_dispatchers()

    def _init_judge_dispatchers(self) -> None:
        if self._contract.invariants and self._contract.invariants.process:
            proc = self._contract.invariants.process
            for jp in proc.judge_predicate:
                self._judge_dispatchers.append(
                    JudgeDispatcher(cost_ceiling=jp.cost_ceiling_usd_per_session, model=jp.model)
                )

    # ------------------------------------------------------------------
    # Drift calibration (opt-in)
    # ------------------------------------------------------------------

    def set_drift_reference(self, distribution: dict[str, float]) -> None:
        """Calibrate the distributional-drift baseline.

        Drift is D(t) = w_c·(1−C) + w_d·JSD(P_t‖P_ref). The JSD (distributional)
        term stays inert until a reference P_ref is set — so without calibration
        only the compliance axis of drift is active (this matches abc's
        measurement-plane default; auto-calibration is a tracked follow-up). Call
        this once after a representative warmup with the expected action/tool
        distribution to activate the distributional term.
        """
        self._drift.set_reference(distribution)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def attach_store(self, store: SessionStore) -> None:
        """Attach a SessionStore and immediately restore persisted state."""
        self._store = store
        self._load_from_store()

    def _load_from_store(self) -> None:
        from agentassert_abc.gateway import serializers

        if self._store is None:
            return
        if data := self._store.get("theta"):
            serializers.load_theta(self._theta, data)
        if data := self._store.get("drift"):
            serializers.load_drift(self._drift, data)
        if data := self._store.get("violations"):
            serializers.load_violations(self._violations, data)
        if data := self._store.get("session_meta"):
            serializers.load_meta(self, data)
        if data := self._store.get("cost"):
            serializers.load_cost(self, data)
        if data := self._store.get("repetition"):
            serializers.load_repetition(self, data)

    def _persist_to_store(self) -> None:
        """Mark all state dirty in the store (no IO — write-behind)."""
        from agentassert_abc.gateway import serializers

        if self._store is None:
            return
        self._store.put("theta", serializers.dump_theta(self._theta))
        self._store.put("drift", serializers.dump_drift(self._drift))
        self._store.put("violations", serializers.dump_violations(self._violations))
        self._store.put("session_meta", serializers.dump_meta(self))
        self._store.put("cost", serializers.dump_cost(self))
        self._store.put("repetition", serializers.dump_repetition(self))

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, event: TypeCEvent) -> DecisionResult:
        with self._lock:
            if isinstance(event, TurnEnd):
                self._turn_count += 1
                self._seen_tools_turn.clear()

            result = dispatch_event(
                event,
                self._compiled,
                self._drift,
                self._theta,
                self._violations,
                seen_session=self._seen_tools_session,
                seen_turn=self._seen_tools_turn,
                accumulated_cost=self._accumulated_cost_usd,
                tool_history=self._tool_call_history,
                seq_hash_counts=self._sequence_hash_counts,
            )

            if isinstance(event, PreAction) and not result.is_deny():
                self._seen_tools_session.add(event.tool)
                self._seen_tools_turn.add(event.tool)
                self._commit_to_history(event.tool)

            if result.is_deny():
                self._deny_count += 1
                self._theta.record_violation()

            self._persist_to_store()
            return result

    def _commit_to_history(self, tool: str) -> None:
        """Add `tool` to repetition-guard history + update sequence-hash counts.

        Skips tools matching `repetition_guard`'s ignore patterns.
        """
        if self._compiled.repetition_guard_config is None:
            return
        for pattern in self._compiled.repetition_guard_ignore_patterns:
            if pattern.search(tool):
                return

        self._tool_call_history.append(tool)
        window_size = self._compiled.repetition_guard_config.window_size
        if len(self._tool_call_history) >= window_size:
            window = tuple(list(self._tool_call_history)[-window_size:])
            seq_key = hashlib.md5("|".join(window).encode()).hexdigest()  # noqa: S324
            self._sequence_hash_counts[seq_key] += 1

    def schedule_judge_evaluation(self, turn_output: str, session_id: str) -> None:
        if not self._contract.invariants or not self._contract.invariants.process:
            return
        proc = self._contract.invariants.process
        for jp_config in proc.judge_predicate:
            for dispatcher in self._judge_dispatchers:
                if dispatcher._model == jp_config.model and dispatcher.should_sample(
                    jp_config.sample_rate
                ):
                    _schedule_judge_task(
                        dispatcher, jp_config, turn_output, session_id, self._theta
                    )
                    break

    def close(self) -> SessionEnd:
        with self._lock:
            self._persist_to_store()
            if self._store is not None:
                self._store.flush()
                self._store.close()
                self._store = None

            theta = self._theta.compute()
            report = self._build_drift_report()
            return SessionEnd(
                session_id=self._contract.name,
                contract_id=self._contract.name,
                theta=theta,
                theta_penalty=self._theta.penalty_sum,
                drift_report=report,
            )

    def _build_drift_report(self) -> DriftReport:
        """Build a DriftReport from abc v2's DriftTracker + local tool history.

        See `gateway/events.py::DriftReport` docstring — abc's DriftTracker
        exposes no public per-tool distribution, so `tool_distribution` is
        computed here from the enforcer's own bounded `_tool_call_history`.
        `current_jsd` uses `mean_drift` (the composite D(t) average, not
        pure JSD) as the closest available session-level summary.
        """
        counts = Counter(self._tool_call_history)
        total = sum(counts.values()) or 1
        return DriftReport(
            current_jsd=self._drift.mean_drift,
            tool_distribution={k: v / total for k, v in counts.items()},
            window_size=self._drift_window,
            violation_count=self._violations.hard_count(),
        )

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def deny_count(self) -> int:
        return self._deny_count

    @classmethod
    def from_yaml(cls, path: str) -> SessionEnforcer:
        """Load a ContractSpecExtended from YAML and build a SessionEnforcer.

        Delegates to the shared DSL parser (Phase D:
        `agentassert_abc.dsl.parser.load_contract_extended`), which applies
        both the base semantic validator and the process-operator semantic
        validator. Any load / parse / validation failure surfaces uniformly
        as `ContractLoadError`.
        """
        try:
            from agentassert_abc.dsl.parser import load_contract_extended
        except ImportError as e:
            raise ContractLoadError(
                "ruamel.yaml is required for YAML parsing. "
                "Install with: pip install agentassert-abc[yaml]"
            ) from e

        from agentassert_abc.exceptions import ContractParseError, ContractValidationError

        try:
            contract = load_contract_extended(path)
        except FileNotFoundError as e:
            raise ContractLoadError(f"Contract file not found: {path}") from e
        except (ContractParseError, ContractValidationError) as e:
            raise ContractLoadError(str(e)) from e

        return cls(contract)


def _schedule_judge_task(
    dispatcher: JudgeDispatcher,
    jp_config: Any,
    turn_output: str,
    session_id: str,
    theta: ThetaScorer,
) -> None:
    async def _run() -> None:
        # Fail-open by design (this is a fire-and-forget background task that
        # must never crash the caller's event loop) — but NOT silent. Unlike
        # typec's bare `except Exception: pass` (port-delta CRIT note), any
        # failure is logged. `apply_penalty` is a real method on abc v2's
        # `ThetaScorer` (Phase B) — silent-break #2 (AttributeError swallowed
        # here in typec) cannot recur.
        try:
            passed, _cost = await dispatcher.evaluate(jp_config.rubric, turn_output, session_id)
            if not passed and jp_config.action_on_fail == "theta_penalty":
                theta.apply_penalty(_JUDGE_FAIL_PENALTY)
        except Exception:
            logger.warning("Judge task failed for session %s", session_id, exc_info=True)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run())
    except RuntimeError:
        asyncio.run(_run())
