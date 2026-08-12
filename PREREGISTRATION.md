# AgentAssert v2 — Confirmatory Experiment Preregistration

**Registered:** 2026-08-10 (git commit timestamp = before any confirmatory outcome).
**Study:** Does model-sharing across compositional AI-agent pipelines induce
*correlated* behavioral-contract failures, violating the independence assumption
in naive compositional reliability bounds (the C5 question)?
**Status:** LOCKED before the confirmatory run. Any change after outcomes are
generated is a dated amendment, disclosed in the paper.

> Note: this internal preregistration is git-timestamped. Before paper
> submission it should also be posted to a public registry (e.g. OSF) for an
> external timestamp.

## 1. Hypotheses (primary, confirmatory)

- **H1 (dependence exists under sharing).** In the `same_model` condition, the
  co-failure dependence between the two pipeline agents is positive:
  Kendall's τ_a > 0, with the bootstrap CI excluding 0.
- **H2 (dependence decreases with less sharing).** τ_a is (weakly) monotone
  decreasing across `same_model ≥ same_vendor ≥ different_vendor`. Rationale:
  more shared inductive bias → more correlated failure modes.
- **H3 (naive bound is anti-conservative under sharing).** Observed graph
  reliability P(Y_G=1) exceeds the independence product Π p_i under positive
  dependence (the compositional gap is signed and non-zero), so a
  dependence-aware bound is required.

Secondary/descriptive: graph e-process certification (final wealth, first
crossing vs p0=0.90) and Jacobi drift per condition. Not used to accept H1–H3.

## 2. Design

- **Task domain:** real retail + financial agent missions (six seeded
  generators, `experiments/domains.py`), scored by **deterministic gold code —
  no LLM judge** (order arithmetic, refund policy, promo cap; transaction limit,
  watchlist screen, mandatory disclaimer). Mission→task is a fixed sha256 hash
  of `mission_id` (reproducible).
- **Motif (topology):** `series2` — a 2-agent handoff A→B; both must satisfy the
  contract for Y_G=1. Concentrates all n on one scored agent pair for the
  tightest dependence estimate. (Other motifs = future/secondary work.)
- **Conditions (5) and sample sizes (LOCKED):**

  | Condition | model_a × model_b | Backend | n (missions) | concurrency |
  |---|---|---|---:|---:|
  | same_model | mistral-small-24b × mistral-small-24b | OpenRouter | 6000 | 16 |
  | same_vendor | mistral-small-24b × ministral-8b | OpenRouter | 6000 | 16 |
  | different_vendor | mistral-small-24b × gemma-3-12b-it | OpenRouter | 6000 | 16 |
  | different_vendor_meta | muse-spark-1.2-contributor × mistral-small-24b | Meta | 2000 | 2 |
  | different_vendor_grok | grok-4.5 × mistral-small-24b | Grok (bridge) | 2000 | 4 |

  Primary confirmatory arms (H1–H3) are the three OpenRouter arms at n=6000
  (full power for a tight τ CI / identifiability diameter). The two breadth
  arms (n=2000) are cross-backend replication of the different_vendor result;
  their smaller n is preregistered here, not chosen post hoc.

- **Roster:** LOCKED (config.py). Model IDs verified present + non-reasoning
  (except Meta/Grok reasoning models handled by effort/anchor). Substitution
  only via a dated amendment before outcomes.

## 3. Frozen sampling (LLD-E §5.1)

`temperature=0.2`, `top_p=1.0`, `max_output_tokens=160`; Meta Spark
`reasoning.effort="minimal"`. Client-side prompt truncation at 3200 chars.
Exactly these values for every call.

## 4. Primary analysis (fixed in advance)

- **Dependence:** `analysis.dependence_report` → Kendall τ_a on the co-failure
  indicators of the two scored agents, with a **cluster bootstrap** CI
  (n_boot=2000, α=0.05) clustered by mission. H1: CI excludes 0. H2: compare
  point τ_a across the three primary arms for monotonicity.
- **Composition:** `composition_report` → observed P(Y_G=1), independence
  product, signed gap (H3).
- Certification (graph e-process) and drift (Jacobi) reported per condition as
  secondary descriptors.

## 5. Stopping / budget (safety, LLD-E §6)

- Hard budget stop **$19.50** (cap $20.00), enforced by the prospective §6.3
  batch gate (armed automatically for paid clients; counts real API calls).
- `FRONTIER_ENABLED` is enabled **in-process only**; the on-disk default stays
  `False`.
- Crash-proof: each arm resumes from its own JSONL; a killed run re-launches
  with the same command and never re-pays completed missions.
- One retry ladder with exponential backoff per call (transient errors only);
  content failures fail fast and are logged per mission, not silently dropped.

## 6. Outcomes storage

Per-arm JSONL at `experiments-baseline/frontier_<condition>.jsonl`
(+ `.progress.json`, `.failures.jsonl`). Analysis reproduces from these logs.

## 7. What would falsify the thesis

If `same_model` τ_a CI includes 0 (no dependence), or τ_a does **not** decrease
from `same_model` to `different_vendor`, or the composition gap is ~0, the C5
"sharing induces correlated failure" claim is **not** supported and the paper
reports that null honestly.
