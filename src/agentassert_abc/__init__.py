# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
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
    ContractLoadError,
    ContractParseError,
    ContractValidationError,
    DriftThresholdError,
    ExprEvaluationError,
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
        # DSL parsing — extended (enforcement plane: invariants.process)
        "load_contract_extended": ("agentassert_abc.dsl.parser", "load_contract_extended"),
        "loads_contract_extended": ("agentassert_abc.dsl.parser", "loads_contract_extended"),
        "validate": ("agentassert_abc.dsl.validator", "validate_contract"),
        # Evaluation
        "evaluate": ("agentassert_abc.evaluator.engine", "evaluate"),
        "evaluate_preconditions": ("agentassert_abc.evaluator.engine", "evaluate_preconditions"),
        # Monitor
        "SessionMonitor": ("agentassert_abc.monitor.session", "SessionMonitor"),
        "compute_theta": ("agentassert_abc.metrics.theta", "compute_theta"),
        "ThetaScorer": ("agentassert_abc.metrics.theta", "ThetaScorer"),
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
        "compose_guarantees_with_conditions": (
            "agentassert_abc.certification.composition",
            "compose_guarantees_with_conditions",
        ),
        "ConditionVerdict": ("agentassert_abc.certification.composition", "ConditionVerdict"),
        "ConditionResult": ("agentassert_abc.certification.composition", "ConditionResult"),
        "CompositionResult": ("agentassert_abc.certification.composition", "CompositionResult"),
        "SatisfactionChecker": (
            "agentassert_abc.certification.satisfaction", "SatisfactionChecker",
        ),
        "SatisfactionVerdict": (
            "agentassert_abc.certification.satisfaction", "SatisfactionVerdict",
        ),
        "SessionLog": ("agentassert_abc.certification.satisfaction", "SessionLog"),
        "TurnRecord": ("agentassert_abc.certification.satisfaction", "TurnRecord"),
        # Metrics (F3/F4: OU dynamics + Lyapunov)
        "OUFitter": ("agentassert_abc.metrics.dynamics", "OUFitter"),
        "LyapunovStabilityCheck": ("agentassert_abc.metrics.dynamics", "LyapunovStabilityCheck"),
        "StabilityVerdict": ("agentassert_abc.metrics.dynamics", "StabilityVerdict"),
        "StabilityReport": ("agentassert_abc.metrics.dynamics", "StabilityReport"),
        "OUParameters": ("agentassert_abc.metrics.dynamics", "OUParameters"),
        # Evaluator (G5: sandboxed expr)
        "SafeExprEvaluator": ("agentassert_abc.evaluator.expr_eval", "SafeExprEvaluator"),
        "ExprResult": ("agentassert_abc.evaluator.expr_eval", "ExprResult"),
        # Adapters (F-08: export adapters)
        "GenericAdapter": ("agentassert_abc.integrations.generic", "GenericAdapter"),
        "LangGraphAdapter": ("agentassert_abc.integrations.langgraph", "LangGraphAdapter"),
        "CrewAIAdapter": ("agentassert_abc.integrations.crewai", "CrewAIAdapter"),
        "OpenAIAgentsAdapter": (
            "agentassert_abc.integrations.openai_agents", "OpenAIAgentsAdapter",
        ),
        "PydanticAIAdapter": ("agentassert_abc.integrations.pydantic_ai", "PydanticAIAdapter"),
        # Metrics (F0/F1: adaptive thresholds)
        "AdaptiveThresholdEngine": ("agentassert_abc.metrics.adaptive", "AdaptiveThresholdEngine"),
        "AdaptiveConfig": ("agentassert_abc.metrics.adaptive", "AdaptiveConfig"),
        # Monitor (F9: EventBus + MCP)
        "EventBus": ("agentassert_abc.monitor.events", "EventBus"),
        "EventKind": ("agentassert_abc.monitor.events", "EventKind"),
        "ViolationEvent": ("agentassert_abc.monitor.events", "ViolationEvent"),
        "RecoveryEvent": ("agentassert_abc.monitor.events", "RecoveryEvent"),
        "DriftWarningEvent": ("agentassert_abc.monitor.events", "DriftWarningEvent"),
        "SessionSummaryEvent": ("agentassert_abc.monitor.events", "SessionSummaryEvent"),
        "MCPServerMonitor": ("agentassert_abc.monitor.mcp_monitor", "MCPServerMonitor"),
        "ToolCallVerdict": ("agentassert_abc.monitor.mcp_monitor", "ToolCallVerdict"),
        # Exporters (Phase 6)
        "OTelExporter": ("agentassert_abc.exporters.otel", "OTelExporter"),
        "OTelSpan": ("agentassert_abc.exporters.otel", "OTelSpan"),
        "EUAIActReportGenerator": (
            "agentassert_abc.exporters.eu_ai_act", "EUAIActReportGenerator",
        ),
        "EUAIReport": ("agentassert_abc.exporters.eu_ai_act", "EUAIReport"),
        # A2A (Phase 7)
        "A2AComplianceBridge": ("agentassert_abc.integrations.a2a", "A2AComplianceBridge"),
        "A2AComplianceResult": ("agentassert_abc.integrations.a2a", "A2AComplianceResult"),
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
    "load_contract_extended",
    "loads_contract_extended",
    "validate",
    "evaluate",
    "evaluate_preconditions",
    "SessionMonitor",
    "compute_theta",
    "ThetaScorer",
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
    "compose_guarantees_with_conditions",
    "ConditionVerdict",
    "ConditionResult",
    "CompositionResult",
    "SatisfactionChecker",
    "SatisfactionVerdict",
    "SessionLog",
    "TurnRecord",
    # Metrics (F3/F4)
    "OUFitter",
    "LyapunovStabilityCheck",
    "StabilityVerdict",
    "StabilityReport",
    "OUParameters",
    # Evaluator (G5)
    "SafeExprEvaluator",
    "ExprResult",
    # Adapters (F-08)
    "GenericAdapter",
    "LangGraphAdapter",
    "CrewAIAdapter",
    "OpenAIAgentsAdapter",
    "PydanticAIAdapter",
    # Metrics (F0/F1: adaptive thresholds)
    "AdaptiveThresholdEngine",
    "AdaptiveConfig",
    # Monitor (F9)
    "EventBus",
    "EventKind",
    "ViolationEvent",
    "RecoveryEvent",
    "DriftWarningEvent",
    "SessionSummaryEvent",
    "MCPServerMonitor",
    "ToolCallVerdict",
    # Exporters (Phase 6)
    "OTelExporter",
    "OTelSpan",
    "EUAIActReportGenerator",
    "EUAIReport",
    # A2A (Phase 7)
    "A2AComplianceBridge",
    "A2AComplianceResult",
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
    "ContractLoadError",
    "ContractParseError",
    "ContractValidationError",
    "DriftThresholdError",
    "ExprEvaluationError",
    "PreconditionFailedError",
    "RecoveryFailedError",
    "StateExtractionError",
]