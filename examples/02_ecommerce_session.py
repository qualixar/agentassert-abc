# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Example 02: E-Commerce Product Recommendation — Full Session.

Demonstrates the patent's primary validation scenario: an e-commerce
product recommendation agent monitored across a multi-turn session.

Uses the contract from TECHNICAL-ATTACHMENT.md §4.1 (gold standard).
"""

from pathlib import Path

import agentassert_abc as aa
from agentassert_abc.exceptions import ContractBreachError
from agentassert_abc.integrations.generic import GenericAdapter

# Load the e-commerce contract
CONTRACT_PATH = Path(__file__).parent.parent / "contracts" / "examples" / "ecommerce-product-recommendation.yaml"

# Fallback to test fixture if contracts dir not yet populated
if not CONTRACT_PATH.exists():
    CONTRACT_PATH = (
        Path(__file__).parent.parent
        / "tests" / "test_dsl" / "fixtures"
        / "ecommerce-product-recommendation.yaml"
    )

contract = aa.load(str(CONTRACT_PATH))
adapter = GenericAdapter(contract)

# Simulate a 5-turn e-commerce session
session_turns = [
    # Turn 1: Customer browses, agent recommends — COMPLIANT
    {
        "session.customer_identified": True,
        "system.catalog_service_status": "available",
        "context.promotions_loaded": True,
        "output.competitor_reference_detected": False,
        "output.unverified_availability_claim": False,
        "output.unauthorized_discount_offered": False,
        "output.pii_detected": False,
        "output.sponsored_items_disclosed": True,
        "output.undisclosed_dynamic_pricing": False,
        "output.brand_tone_score": 0.92,
        "output.recommendation_relevance_score": 0.88,
        "output.completeness_score": 0.85,
        "output.upsell_count": 1,
        "response.latency_ms": 800,
        "output.customer_satisfaction_score": 0.78,
        "tools.all_calls_authorized": True,
        "session.total_tokens": 2500,
        "session.total_cost_usd": 0.15,
    },
    # Turn 2: Agent upsells too much — SOFT VIOLATION
    {
        "session.customer_identified": True,
        "system.catalog_service_status": "available",
        "context.promotions_loaded": True,
        "output.competitor_reference_detected": False,
        "output.unverified_availability_claim": False,
        "output.unauthorized_discount_offered": False,
        "output.pii_detected": False,
        "output.sponsored_items_disclosed": True,
        "output.undisclosed_dynamic_pricing": False,
        "output.brand_tone_score": 0.75,
        "output.recommendation_relevance_score": 0.65,
        "output.completeness_score": 0.7,
        "output.upsell_count": 4,  # Exceeds lte: 2
        "response.latency_ms": 1500,
        "output.customer_satisfaction_score": 0.5,
        "tools.all_calls_authorized": True,
        "session.total_tokens": 6000,
        "session.total_cost_usd": 0.35,
    },
    # Turn 3: Agent recovers after correction — COMPLIANT
    {
        "session.customer_identified": True,
        "system.catalog_service_status": "available",
        "context.promotions_loaded": True,
        "output.competitor_reference_detected": False,
        "output.unverified_availability_claim": False,
        "output.unauthorized_discount_offered": False,
        "output.pii_detected": False,
        "output.sponsored_items_disclosed": True,
        "output.undisclosed_dynamic_pricing": False,
        "output.brand_tone_score": 0.88,
        "output.recommendation_relevance_score": 0.82,
        "output.completeness_score": 0.9,
        "output.upsell_count": 2,
        "response.latency_ms": 900,
        "output.customer_satisfaction_score": 0.72,
        "tools.all_calls_authorized": True,
        "session.total_tokens": 9000,
        "session.total_cost_usd": 0.50,
    },
    # Turn 4: Slow response — SOFT VIOLATION
    {
        "session.customer_identified": True,
        "system.catalog_service_status": "available",
        "context.promotions_loaded": True,
        "output.competitor_reference_detected": False,
        "output.unverified_availability_claim": False,
        "output.unauthorized_discount_offered": False,
        "output.pii_detected": False,
        "output.sponsored_items_disclosed": True,
        "output.undisclosed_dynamic_pricing": False,
        "output.brand_tone_score": 0.80,
        "output.recommendation_relevance_score": 0.75,
        "output.completeness_score": 0.6,
        "output.upsell_count": 1,
        "response.latency_ms": 4500,  # Exceeds lte: 3000
        "output.customer_satisfaction_score": 0.55,
        "tools.all_calls_authorized": True,
        "session.total_tokens": 15000,
        "session.total_cost_usd": 0.90,
    },
    # Turn 5: PII leak attempt — HARD VIOLATION
    {
        "session.customer_identified": True,
        "system.catalog_service_status": "available",
        "context.promotions_loaded": True,
        "output.competitor_reference_detected": False,
        "output.unverified_availability_claim": False,
        "output.unauthorized_discount_offered": False,
        "output.pii_detected": True,  # HARD VIOLATION
        "output.sponsored_items_disclosed": True,
        "output.undisclosed_dynamic_pricing": False,
        "output.brand_tone_score": 0.85,
        "output.recommendation_relevance_score": 0.80,
        "output.completeness_score": 0.8,
        "output.upsell_count": 1,
        "response.latency_ms": 1000,
        "output.customer_satisfaction_score": 0.65,
        "tools.all_calls_authorized": True,
        "session.total_tokens": 18000,
        "session.total_cost_usd": 1.10,
    },
]

print("=" * 60)
print("AgentAssert — E-Commerce Session Demo (Patent §4.1)")
print("=" * 60)

for i, turn in enumerate(session_turns):
    try:
        result = adapter.check_and_raise(turn)
        status = "COMPLIANT" if result.soft_violations == 0 else "SOFT VIOLATION"
        print(f"\nTurn {i + 1}: {status}")
        if result.soft_violations > 0:
            print(f"  Soft violations: {result.violated_names}")
            print(f"  Recovery needed: {result.recovery_needed}")
    except ContractBreachError as e:
        print(f"\nTurn {i + 1}: HARD BREACH")
        print(f"  {e}")
        print("  Action: Response blocked, escalation triggered")

# Session metrics
summary = adapter.session_summary()
print("\n" + "=" * 60)
print("Session Summary")
print("=" * 60)
print(f"  Total turns: {summary.turn_count}")
print(f"  Hard violations: {summary.total_hard_violations}")
print(f"  Soft violations: {summary.total_soft_violations}")
print(f"  Reliability Index (Θ): {summary.theta:.3f}")
print(f"  Mean drift: {summary.mean_drift:.3f}")
print(f"  Recovery rate: {summary.recovery_rate:.2f}")
verdict = "DEPLOY" if summary.theta >= 0.90 else "DO NOT DEPLOY"
print(f"  Deployment decision: {verdict}")
