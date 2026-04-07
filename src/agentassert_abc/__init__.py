# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""AgentAssert: Formal behavioral specification and runtime enforcement for AI agents.

Agent Behavioral Contracts (ABC) — the only framework combining all 6 pillars:
1. ContractSpec DSL (YAML-based behavioral specification)
2. Hard/Soft constraint separation with recovery
3. JSD-based drift detection
4. (p, delta, k)-satisfaction guarantees
5. Compositional safety proofs for multi-agent pipelines
6. Ornstein-Uhlenbeck drift dynamics with Lyapunov stability

Usage:
    import agentassert_abc as aa

    contract = aa.load("contract.yaml")
    monitor = aa.SessionMonitor(contract)
    result = monitor.step(agent_state)
    summary = monitor.session_summary()

Paper: https://arxiv.org/abs/2602.22302
Website: https://agentassert.com
"""

from agentassert_abc._version import __version__

# Exceptions
from agentassert_abc.exceptions import (
    AgentAssertError,
    ContractBreachError,
    ContractParseError,
    ContractValidationError,
    DriftThresholdError,
    PreconditionFailedError,
    RecoveryFailedError,
    StateExtractionError,
)

# Core models
from agentassert_abc.models import (
    ConstraintCheck,
    ContractMetadata,
    ContractSpec,
    DriftConfig,
    DriftThresholds,
    DriftWeights,
    Governance,
    GovernanceConstraint,
    HardConstraint,
    Invariants,
    Precondition,
    RecoveryAction,
    RecoveryConfig,
    ReliabilityConfig,
    ReliabilityWeights,
    SatisfactionParams,
    SoftConstraint,
)


def __getattr__(name: str):  # type: ignore[no-untyped-def]  # noqa: N807
    """Lazy imports for heavy modules (YAML parser, monitor, metrics).

    This keeps `import agentassert_abc` fast (<10ms) by deferring
    scipy/ruamel.yaml imports until actually needed.
    L-06: Caches imported functions in module globals for repeated access.
    """
    _lazy_map = {
        # DSL parsing
        "load": ("agentassert_abc.dsl.parser", "load_contract"),
        "loads": ("agentassert_abc.dsl.parser", "loads_contract"),
        "parse": ("agentassert_abc.dsl.parser", "parse_contract"),
        "parses": ("agentassert_abc.dsl.parser", "parses_contract"),
        "validate": ("agentassert_abc.dsl.validator", "validate_contract"),
        # Evaluation
        "evaluate": ("agentassert_abc.evaluator.engine", "evaluate"),
        "evaluate_preconditions": ("agentassert_abc.evaluator.engine", "evaluate_preconditions"),
        # Monitor
        "SessionMonitor": ("agentassert_abc.monitor.session", "SessionMonitor"),
        "compute_theta": ("agentassert_abc.metrics.theta", "compute_theta"),
        # Result types (F-06: export all result types)
        "StepResult": ("agentassert_abc.monitor.models", "StepResult"),
        "SessionSummary": ("agentassert_abc.monitor.models", "SessionSummary"),
        "PreconditionCheckResult": ("agentassert_abc.monitor.models", "PreconditionCheckResult"),
        "EvaluationResult": ("agentassert_abc.evaluator.models", "EvaluationResult"),
        "ConstraintResult": ("agentassert_abc.evaluator.models", "ConstraintResult"),
        "ParseResult": ("agentassert_abc.dsl.models", "ParseResult"),
        # Certification (F-07: export certification)
        "SPRTCertifier": ("agentassert_abc.certification.sprt", "SPRTCertifier"),
        "compose_guarantees": ("agentassert_abc.certification.composition", "compose_guarantees"),
        # Adapters (F-08: export adapters)
        "GenericAdapter": ("agentassert_abc.integrations.generic", "GenericAdapter"),
    }
    if name in _lazy_map:
        import importlib

        module_path, attr_name = _lazy_map[name]
        module = importlib.import_module(module_path)
        obj = getattr(module, attr_name)
        globals()[name] = obj  # Cache for next access
        return obj
    raise AttributeError(f"module 'agentassert_abc' has no attribute {name!r}")


__all__ = [
    "__version__",
    # Convenience API (lazy-loaded)
    "load",
    "loads",
    "parse",
    "parses",
    "validate",
    "evaluate",
    "evaluate_preconditions",
    "SessionMonitor",
    "compute_theta",
    # Result types (F-06)
    "StepResult",
    "SessionSummary",
    "PreconditionCheckResult",
    "EvaluationResult",
    "ConstraintResult",
    "ParseResult",
    # Certification (F-07)
    "SPRTCertifier",
    "compose_guarantees",
    # Adapters (F-08)
    "GenericAdapter",
    # Models
    "ConstraintCheck",
    "ContractMetadata",
    "ContractSpec",
    "DriftConfig",
    "DriftThresholds",
    "DriftWeights",
    "Governance",
    "GovernanceConstraint",
    "HardConstraint",
    "Invariants",
    "Precondition",
    "RecoveryAction",
    "RecoveryConfig",
    "ReliabilityConfig",
    "ReliabilityWeights",
    "SatisfactionParams",
    "SoftConstraint",
    # Exceptions
    "AgentAssertError",
    "ContractBreachError",
    "ContractParseError",
    "ContractValidationError",
    "DriftThresholdError",
    "PreconditionFailedError",
    "RecoveryFailedError",
    "StateExtractionError",
]
