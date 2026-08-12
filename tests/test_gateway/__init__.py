# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Gateway (Type C consolidation) test package.

Tests migrated from `agentassert-typec` (v0.6.2, MIT) per the migration notes
(the migration notes). Import
paths point at `agentassert_abc.gateway`/`agentassert_abc.process` instead
of the discontinued `agentassert_typec_core` package. Internal naming
(`tap_*`, `mcp__hermes*`, `ds-flash-free`) has been genericized throughout.

NOT migrated (out of scope for this port — see the migration scope):
- DSL parser/validator tests (`test_parser.py`, `test_validator.py`,
  parts of `test_precise_gaps.py`/`test_final_gaps.py`/`test_coverage_gaps.py`)
  — Phase D (DSL extension) is a separate, not-yet-ported phase.
- typec's own discarded `DriftTracker` unit tests (`test_drift.py`) — that
  tracker is discarded entirely; abc v2's `DriftTracker` already has its
  own coverage under `tests/test_metrics/test_drift.py`.
- typec's soft-import `evaluate_abc_check` shim tests — the shim itself
  was discarded in favor of a direct intra-package import.
- OTel exporter tests from `test_otel_judge.py` — merging typec's spans
  into abc's existing `OTelExporter` is Phase F, not part of this port.
- `test_db_isolation_by_session_id` from `test_persistence.py` — exercises
  the typec proxy package (Phase E), not ported here.
"""
