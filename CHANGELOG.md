# CHANGELOG

All notable changes to `agentassert-abc` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [0.6.0] — unreleased

### Added

- **`dependence.jaccard()`** — the model-free failure-set overlap
  `n11 / (n11 + n10 + n01)` on a `CoFailureTable`, the statistic the paper leads
  its co-failure result with. Raises rather than returning `0.0` when neither
  agent failed, since an empty failure union makes the statistic undefined
  ("no overlap" and "no failures observed" are different claims).
- **Certification over arbitrary co-execution moment sets** — `moment_subsets`,
  `empirical_subset_moments`, `moment_lp_all_success_bounds`, and
  `moment_cp_box_floor` in `certification/lp_bound.py`. The Tier-1 floor
  previously accepted only per-stage and pairwise success rates; it now accepts
  any collection of co-execution moments, including triples and higher. On the
  four-stage arm this raises the certified floor from 0.246 to 0.412. Two
  confidence-budget allocations are supported: spend it over the moments
  actually used (tightest), or pre-allocate over a larger family via
  `budget_orders` so that adding a moment can never loosen the result.
  `pairwise_lp_all_success_bounds` and `pairwise_cp_box_floor` are unchanged and
  reproduce bit-for-bit as the order-≤2 case.

- **Coverage-collapse simulation** (`benchmarks/coverage_collapse_sim.py`) — the
  experiment behind the paper's identification result, which the text described
  as reproducible but which never shipped. Defaults reproduce the published
  settings; flags trade fidelity for runtime and every run prints its parameters.

### Fixed

- **The proxy and Claude Code hook now populate constraint state.** Both left
  invariant fields unset — the proxy sent only a response byte count, the hook
  sent nothing — so every semantic invariant evaluated false and a fully
  compliant agent was scored at zero compliance with a violation recorded each
  turn. Both now flatten the response into the documented field convention, and
  refuse at load time any contract whose fields that surface can never supply,
  rather than silently reporting the agent as failing.
- **Distributional drift is measured by default.** The reference distribution
  was never established anywhere in the package, so the divergence half of the
  drift score — 40% of its weight — was permanently zero for every user. The
  baseline is now adopted automatically after a configurable number of turns
  (`auto_calibrate_after`, default 10; 0 restores the previous opt-in
  behaviour). Setting a reference explicitly still takes precedence.

### Changed

- **Drift stability and admissibility are now separate verdicts.** Mean
  reversion is judged on the reversion rate alone; where the process settles is
  judged separately against a configurable critical drift level and reported as
  the new `StabilityVerdict.INADMISSIBLE`. `DIVERGENT` is now reserved for
  having no restoring force at all. `StabilityReport` gains `stable`,
  `admissible`, `d_star`, and `d_crit`.
  **Breaking:** sequences that previously reported `DIVERGENT` because the
  natural drift rate exceeded the reversion rate now report `INADMISSIBLE`.
  That comparison comes from the withdrawn v1 formulation; it is not
  scale-invariant, because rescaling the drift score rescales one side of it
  and not the other.

### Fixed

- **Documentation accuracy** — the 0.4.0 entry below listed Jaccard among the
  shipped dependence estimators, but no such function existed in the package.
  The statistic is now actually implemented and exported, with tests pinning it
  against the published Table 1 values.

## [0.5.0] — 2026-08-11

### Added

- **Runtime enforcement plane** — real-time behavioral-contract enforcement that returns
  ALLOW / DENY / REDACT / MODIFY decisions on tool calls and model responses, complementing
  the existing measurement and certification planes. New `agentassert_abc.gateway`
  (`SessionEnforcer`, event dispatch, compiled process invariants, SQLite session persistence).
- **Process invariants** — authored under `invariants.process` in a contract:
  `must_precede`, `must_state`, tool allow/block lists, per-turn context-token budget,
  process-drift guard, sampled LLM-as-judge predicate, PII filter, cost ceiling, and
  repetition guard.
- **Zero-code-change adoption surfaces** (optional install extras):
  - `agentassert-abc[sdk]` — `wrap(client, contract)` one-liner for Anthropic and OpenAI clients.
  - `agentassert-abc[proxy]` — HTTP proxy that enforces contracts across Anthropic / OpenAI /
    Gemini / OpenRouter traffic.
  - `agentassert-abc[claude-code]` — Claude Code hook that enforces a contract on tool use.
- `ContractSpecExtended` and `load_contract_extended` / `loads_contract_extended` — contract
  loading with the process plane and full semantic validation.
- OpenTelemetry spans for enforcement events.

### Changed

- `ContractBreachError` now carries optional structured fields (violation, tool, session,
  decision) while remaining backward-compatible with a plain-message raise.

### Notes

- Consolidates the former `agentassert-typec` packages into `agentassert-abc`. The
  `agentassert-typec-*` distributions continue to work as thin deprecation shims that forward
  here; install the matching extra (`agentassert-abc[gateway]`) going forward.

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
  <br>*Correction (0.6.0): Jaccard was named here but never shipped in 0.4.0; it landed in
  0.6.0. The other estimators in this line were present as described.*
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
