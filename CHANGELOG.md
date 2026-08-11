# CHANGELOG

All notable changes to `agentassert-abc` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-08-11

### Added

- **Dependence-aware compositional reliability certificate** — a tiered certificate that
  does **not** assume stage-failure independence (condition C5):
  - **Tier 0** — exact Clopper–Pearson lower bound on the directly observed all-success /
    κ-of-m quorum event, plus a design-effect-adjusted floor
    (`certification/observed_floor.py`).
  - **Tier 1** — copula-agnostic linear-program bound over the Fréchet identification set
    that tightens as co-execution moments are supplied (`certification/lp_bound.py`).
  - **Tier 2** — Slepian monotone-corner Gaussian model floor, retained as a **diagnostic
    only** (`certification/slepian_floor.py`).
- **Dependence estimators** — failure-set overlap (Jaccard), Kendall τ_a with its ceiling
  ratio, tetrachoric correlation, and a co-failure table (`dependence/estimators.py`), with
  a bootstrap-CI module.
- **Graph e-process certification** — anytime-valid sequential certification
  (`certification/eprocess.py`) and factor-reliability machinery
  (`certification/factor_reliability.py`).
- **Jacobi bounded-drift analysis** in the metrics layer.
- **Experiment harness** — a bounded-concurrency, budget-gated runner (~7× faster) with real
  retail and financial domain missions, a cross-backend `RoutingClient`, and a
  preregistration (`PREREGISTRATION.md`).
- **Results dashboard** (`dashboard/`) — self-contained HTML view of composition,
  certification, and dependence results.

### Changed

- Hardened contracts, certification, drift, and experiment-safety paths.

### Fixed

- Corrected the temporal direction of the C5 (stage-failure independence) check.

### Notes

- Companion paper (v2): *Agent Behavioral Contracts II: Certifying Compositional Reliability
  Without Assuming Independence* — Zenodo DOI **10.5281/zenodo.21888041**. The v1 framework
  and its 1,980-session evaluation remain at **arXiv:2602.22302**. With the v1 patent claim
  withdrawn, all formulas are now disclosed.

## [0.3.0] — 2026-05-24

### Added

- **Adaptive Threshold Engine** — Learns drift thresholds from calibration data.
- **EventBus** — Typed, thread-safe pub/sub with violation, recovery, drift,
  and session summary events.
- **MCP Server Monitor** — Enforces contracts on MCP tool calls at pre-invoke
  and post-invoke stages.
- **Framework adapters** — PydanticAIAdapter and A2A compliance bridge.
- **OTel Exporter** — OpenTelemetry-span compatibility layer.
- **EU AI Act Report Generator** — Article 12/14/15 compliance evidence.
- **Visual Dashboard** — Self-contained dashboard with Theta gauge, drift
  trajectory, compliance bars, and violation timeline.
- **F2 (p, δ, k)-Satisfaction session-level check** — New module
  `agentassert_abc.certification.satisfaction`. `SatisfactionChecker` computes
  the three F2 conditions on a session log: hard-compliance probability (p),
  max soft deviation (δ), and recovery window (k).
- **F3 OU dynamics + F4 Lyapunov stability verdict** — New module
  `agentassert_abc.metrics.dynamics`. `OUFitter` performs maximum-likelihood
  fit of (α, γ, σ) to observed drift sequences. `LyapunovStabilityCheck`
  returns CONVERGENT / DIVERGENT / INCONCLUSIVE based on V(e) decay analysis.
- **F5 C1-C5 composition condition checkers** — Extended
  `agentassert_abc.certification.composition` with `check_c1_type_compatibility`,
  `check_c2_invariant_preservation`, `check_c3_monotone_drift`,
  `check_c4_recovery_propagation`, `check_c5_independence`, and
  `compose_guarantees_with_conditions(...)`.
- **`expr` operator** — New module `agentassert_abc.evaluator.expr_eval`.
  All 14 ContractSpec operators operational.
- **Wired exceptions** — `DriftThresholdError`, `RecoveryFailedError`, and
  `PreconditionFailedError` now raise at appropriate runtime call sites.

### Changed

- `SessionMonitor` accepts optional `raise_on_drift`, `drift_threshold`,
  and `max_recovery_attempts`.
- Public API expanded to 70+ exports.

### Backward Compatibility

- `compose_guarantees(p_a, p_b, p_h)` signature and return value UNCHANGED.
- All previously working operators continue to work identically.

## [0.2.3] — 2026-04-07

### Fixed

- PyPI sdist exclusions updated.

## [0.2.2] — 2026-04-06

### Fixed

- Install command corrected in README.

## [0.2.1] — 2026-04-05

### Added

- Qualixar platform context section in README.

## [0.2.0] — 2026-04-04

### Changed

- License migration: Elastic-2.0 → AGPL-3.0-or-later.

## [0.1.0] — 2026-02-25

### Added

- Initial release accompanying paper [arXiv:2602.22302](https://arxiv.org/abs/2602.22302).
- Six pillars implementation:
  - ContractSpec DSL parser (YAML)
  - 14 ContractSpec operators
  - Hard/soft constraint evaluator
  - Compliance metric (C_hard, C_soft)
  - JSD-based drift detection
  - SPRT certification
  - Compositional guarantees
  - Reliability Index Θ
- Adapters: GenericAdapter, LangGraphAdapter, CrewAIAdapter, OpenAIAgentsAdapter
- AgentContract-Bench: 293 scenarios, 12 domains
- 12 production contracts in `contracts/examples/`
