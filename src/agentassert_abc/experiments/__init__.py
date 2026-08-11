# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""The $20-capped empirical validation harness (LLD-E).

Runs cheap multi-agent missions (free local models by default; a gated frontier
slice) over contract-bound motifs, logs per-mission outcomes, and feeds the
dependence / e-process / Jacobi analysis. SAFETY: no paid API call happens
unless ``config.FRONTIER_ENABLED`` is explicitly True and the budget permits.
"""
