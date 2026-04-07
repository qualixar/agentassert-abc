# Metrics & Certification

AgentAssert provides four layers of quantitative measurement: compliance tracking, drift detection, a reliability index (Theta), and statistical certification (SPRT). Together, these answer the question: *Is this agent safe to deploy?*

!!! info "Mathematical details"
    This page explains **what** each metric measures and **how to interpret** it. The exact mathematical formulations -- including all equations, proofs, and derivations -- are in the research paper ([arXiv:2602.22302](https://arxiv.org/abs/2602.22302)). The formulas are protected intellectual property.

---

## Compliance Tracking

**What it measures:** The fraction of turns where the agent met all constraints.

AgentAssert tracks compliance separately for hard and soft constraints:

- **Hard compliance** -- Fraction of turns with zero hard violations. A single hard violation in a session is a serious event.
- **Soft compliance** -- Fraction of turns with zero soft violations (after accounting for recovery windows).

```python
summary = adapter.session_summary()

print(f"Hard compliance: {summary.mean_c_hard:.2%}")  # e.g., 100.00%
print(f"Soft compliance: {summary.mean_c_soft:.2%}")   # e.g., 93.33%
```

### Satisfaction Guarantees

The contract's `satisfaction` section defines a compliance target using three parameters:

- **p** -- Target compliance probability (e.g., 0.95 means 95% of turns should be compliant)
- **delta** -- Tolerance band around the target
- **k** -- Window size for rolling evaluation

AgentAssert evaluates compliance over rolling windows of `k` turns and checks whether the observed rate meets the `(p, delta)` target. This provides a probabilistic guarantee -- not just "were all turns compliant?" but "is the agent *reliably* compliant over time?"

The formal definition of (p, delta, k)-satisfaction is in the paper, Section 4.

---

## Drift Detection

**What it measures:** Whether agent behavior is shifting over time, even if individual turns are still compliant.

Consider an agent that starts with a tone score of 0.9 and gradually drops to 0.72 over 50 turns. Every turn passes the `gte: 0.7` constraint, but the trend is concerning. Drift detection catches this.

AgentAssert computes drift using two signals:

1. **Compliance drift** -- Are compliance rates declining over time?
2. **Distributional drift** -- Is the *distribution* of output scores changing? This uses Jensen-Shannon Divergence (JSD), a symmetric measure of how different two probability distributions are.

The two signals are combined using configurable weights:

```yaml
drift:
  weights:
    compliance: 0.6
    distributional: 0.4
  window: 50
  thresholds:
    warning: 0.3
    critical: 0.6
```

### Interpreting Drift Scores

| Drift Score | Interpretation |
|-------------|---------------|
| 0.0 - 0.1 | Stable -- agent behavior is consistent |
| 0.1 - 0.3 | Minor drift -- worth monitoring |
| 0.3 - 0.6 | Warning -- behavioral shift detected |
| 0.6+ | Critical -- significant behavioral change, investigate immediately |

```python
result = adapter.check(agent_output)
print(f"Current drift: {result.drift_score:.3f}")
```

---

## Reliability Index Theta

**What it measures:** A single number between 0 and 1 that summarizes an agent's overall reliability.

Theta combines four components into one score:

| Component | What It Captures |
|-----------|-----------------|
| **Compliance** | How often the agent meets all constraints |
| **Drift** | How stable the agent's behavior is over time |
| **Recovery** | How effectively the agent recovers from soft violations |
| **Stress** | How the agent performs under adversarial or edge-case conditions |

Each component is weighted according to the contract's `reliability.weights` configuration:

```yaml
reliability:
  weights:
    compliance: 0.35
    drift: 0.25
    recovery: 0.20
    stress: 0.20
  deployment_threshold: 0.90
```

### Interpreting Theta

| Theta | Interpretation |
|-------|---------------|
| 0.95+ | Excellent -- production-ready with high confidence |
| 0.90 - 0.95 | Good -- meets typical deployment threshold |
| 0.80 - 0.90 | Marginal -- may need improvement before deployment |
| Below 0.80 | Poor -- not ready for production |

```python
summary = adapter.session_summary()
print(f"Theta: {summary.theta:.3f}")

if summary.theta >= 0.90:
    print("Agent is deployment-ready")
else:
    print("Agent needs improvement")
```

### Computing Theta Directly

You can also compute Theta outside of a session:

```python
from agentassert_abc import compute_theta

theta = compute_theta(
    compliance=0.96,
    drift=0.05,
    recovery=0.90,
    stress=0.85,
    weights={"compliance": 0.35, "drift": 0.25, "recovery": 0.20, "stress": 0.20},
)
print(f"Theta: {theta:.3f}")
```

---

## SPRT Certification

**What it measures:** Whether an agent meets its compliance target with statistical confidence, using the minimum number of test sessions.

Traditional certification requires a fixed number of test sessions (often hundreds). SPRT (Sequential Probability Ratio Test) is a statistical method that reaches a decision -- accept or reject -- as soon as enough evidence has accumulated. In practice, this means 50-80% fewer test sessions.

### How It Works (Conceptually)

1. Define two hypotheses: the agent meets the target (accept) vs. it does not (reject)
2. After each test session, update the evidence ratio
3. If the ratio crosses the acceptance boundary, certify the agent
4. If the ratio crosses the rejection boundary, fail the agent
5. If neither boundary is crossed, continue testing

### Usage

```python
from agentassert_abc.certification.sprt import SPRTCertifier, SPRTDecision

certifier = SPRTCertifier(
    p0=0.85,    # Null hypothesis: true compliance rate
    p1=0.95,    # Alternative hypothesis: target compliance rate
    alpha=0.05,  # Type I error rate (false accept)
    beta=0.10,   # Type II error rate (false reject)
)

# Feed session results one at a time
session_results = [True, True, True, False, True, True, True, True, True, True]

for passed in session_results:
    result = certifier.update(passed)

    if result.decision == SPRTDecision.ACCEPT:
        print(f"CERTIFIED after {result.sessions_used} sessions")
        break
    elif result.decision == SPRTDecision.REJECT:
        print(f"REJECTED after {result.sessions_used} sessions")
        break
    else:
        print(f"Session {result.sessions_used}: continue testing...")
```

### SPRT Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `p0` | Baseline compliance rate (null hypothesis) | 0.85 |
| `p1` | Target compliance rate (alternative hypothesis) | 0.95 |
| `alpha` | Probability of falsely accepting a bad agent | 0.05 |
| `beta` | Probability of falsely rejecting a good agent | 0.10 |

!!! tip "Choosing p0 and p1"
    Set `p0` to the minimum acceptable compliance rate and `p1` to your target. The wider the gap between `p0` and `p1`, the fewer sessions SPRT needs to reach a decision. A gap of 0.10 (e.g., 0.85 to 0.95) is a reasonable default.

---

## Compositional Guarantees

**What it measures:** The minimum compliance rate of a multi-agent pipeline, given the compliance rates of individual agents and the reliability of their handoffs.

When Agent A hands off to Agent B, the pipeline's compliance depends on three factors:

1. Agent A's compliance rate
2. Agent B's compliance rate
3. The reliability of the handoff between them

AgentAssert provides a lower bound on the pipeline's overall compliance:

```python
from agentassert_abc.certification.composition import compose_guarantees

# Agent A: 95% compliant, Agent B: 98% compliant, Handoff: 99% reliable
bound = compose_guarantees(p_a=0.95, p_b=0.98, p_h=0.99)
print(f"Pipeline lower bound: {bound:.3f}")  # >= 0.921
```

This means that if each agent individually meets its contract and the handoff is reliable, the pipeline as a whole is guaranteed to be at least 92.1% compliant.

### Chaining Multiple Agents

For pipelines with more than two agents, compose pairwise:

```python
# A -> B -> C pipeline
bound_ab = compose_guarantees(p_a=0.95, p_b=0.98, p_h=0.99)
bound_abc = compose_guarantees(p_a=bound_ab, p_b=0.97, p_h=0.99)
print(f"Three-agent pipeline bound: {bound_abc:.3f}")
```

!!! note "Conservative bounds"
    Compositional guarantees are lower bounds -- the actual pipeline compliance rate may be higher. This is by design: a guarantee that is sometimes too optimistic is not a guarantee.

---

## Putting It All Together

A typical production workflow:

1. **Develop** -- Write a contract, plug in your adapter, iterate on constraints
2. **Monitor** -- Run the agent in staging, track Theta and drift over sessions
3. **Certify** -- Use SPRT to statistically certify the agent meets the compliance target
4. **Deploy** -- Deploy with runtime monitoring, alert on drift threshold crossings
5. **Compose** -- For multi-agent pipelines, compute compositional bounds

```python
import agentassert_abc as aa
from agentassert_abc.integrations.generic import GenericAdapter
from agentassert_abc.certification.sprt import SPRTCertifier, SPRTDecision

# 1. Load contract and run sessions
contract = aa.load("contracts/examples/customer-support.yaml")

certifier = SPRTCertifier(p0=0.85, p1=0.95, alpha=0.05, beta=0.10)

for session_data in test_sessions:
    adapter = GenericAdapter(contract)

    for turn in session_data:
        adapter.check(turn)

    summary = adapter.session_summary()
    passed = summary.theta >= 0.90

    result = certifier.update(passed)
    if result.decision != SPRTDecision.CONTINUE:
        print(f"Decision: {result.decision.value} after {result.sessions_used} sessions")
        break
```

For the full mathematical treatment of all metrics, see the paper: [arXiv:2602.22302](https://arxiv.org/abs/2602.22302).
