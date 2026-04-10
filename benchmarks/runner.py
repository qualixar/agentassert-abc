# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""AgentContract-Bench v2 — Benchmark Runner.

Loads benchmark scenarios (YAML), evaluates them against domain contracts,
and reports precision/recall of violation detection.

Usage:
    python -m benchmarks.runner                     # Run all scenarios
    python -m benchmarks.runner --domain ecommerce  # Run one domain
    python -m benchmarks.runner --verbose            # Show per-scenario detail
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

import agentassert_abc as aa
from agentassert_abc.evaluator.engine import evaluate

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
CONTRACTS_DIR = Path(__file__).parent.parent / "contracts" / "examples"

yaml = YAML(typ="safe")


@dataclass(frozen=True)
class ScenarioResult:
    """Result of running a single benchmark scenario."""

    scenario_id: str
    domain: str
    passed: bool
    expected_hard: int
    actual_hard: int
    expected_soft: int
    actual_soft: int
    expected_verdict: str
    actual_verdict: str
    detail: str = ""


@dataclass
class DomainSummary:
    """Aggregated results for one domain."""

    domain: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    hard_tp: int = 0
    hard_fp: int = 0
    hard_fn: int = 0
    soft_tp: int = 0
    soft_fp: int = 0
    soft_fn: int = 0


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    total_scenarios: int = 0
    total_passed: int = 0
    total_failed: int = 0
    domains: dict[str, DomainSummary] = field(default_factory=dict)
    failures: list[ScenarioResult] = field(default_factory=list)


def load_scenario(path: Path) -> dict[str, Any]:
    """Load a single benchmark scenario YAML."""
    with path.open() as f:
        return yaml.load(f)


def load_contract_for_domain(domain: str, contract_name: str) -> aa.ContractSpec:
    """Load the domain contract YAML."""
    contract_path = CONTRACTS_DIR / f"{contract_name}.yaml"
    if not contract_path.exists():
        msg = f"Contract not found: {contract_path}"
        raise FileNotFoundError(msg)
    return aa.load(str(contract_path))


def run_scenario(
    scenario: dict[str, Any], contract: aa.ContractSpec
) -> ScenarioResult:
    """Run a single benchmark scenario against its contract."""
    scenario_id = scenario["id"]
    domain = scenario["domain"]
    input_state = scenario["input_state"]
    expected = scenario["expected"]

    expected_hard = expected.get("hard_violations", 0)
    expected_soft = expected.get("soft_violations", 0)
    expected_verdict = expected.get("verdict", "compliant")

    result = evaluate(contract, input_state)

    actual_hard = len(result.hard_violations)
    actual_soft = len(result.soft_violations)

    if actual_hard > 0:
        actual_verdict = "hard_breach"
    elif actual_soft > 0:
        actual_verdict = "soft_violation"
    else:
        actual_verdict = "compliant"

    passed = (
        actual_hard == expected_hard
        and actual_soft == expected_soft
        and actual_verdict == expected_verdict
    )

    detail = ""
    if not passed:
        detail = (
            f"Expected hard={expected_hard} soft={expected_soft} "
            f"verdict={expected_verdict}; "
            f"Got hard={actual_hard} soft={actual_soft} "
            f"verdict={actual_verdict}"
        )

    return ScenarioResult(
        scenario_id=scenario_id,
        domain=domain,
        passed=passed,
        expected_hard=expected_hard,
        actual_hard=actual_hard,
        expected_soft=expected_soft,
        actual_soft=actual_soft,
        expected_verdict=expected_verdict,
        actual_verdict=actual_verdict,
        detail=detail,
    )


def run_domain(
    domain_dir: Path, contract: aa.ContractSpec
) -> list[ScenarioResult]:
    """Run all scenarios in a domain directory."""
    results: list[ScenarioResult] = []
    for scenario_path in sorted(domain_dir.glob("*.yaml")):
        try:
            scenario = load_scenario(scenario_path)
            results.append(run_scenario(scenario, contract))
        except Exception as e:  # noqa: BLE001
            results.append(ScenarioResult(
                scenario_id=scenario_path.stem,
                domain=domain_dir.name,
                passed=False,
                expected_hard=0,
                actual_hard=0,
                expected_soft=0,
                actual_soft=0,
                expected_verdict="unknown",
                actual_verdict="error",
                detail=f"Error loading/running: {e}",
            ))
    return results


def run_all(
    domain_filter: str | None = None, verbose: bool = False
) -> BenchmarkReport:
    """Run the full AgentContract-Bench suite."""
    report = BenchmarkReport()

    # Map domain dirs to contract names
    domain_contract_map: dict[str, str] = {
        "ecommerce": "ecommerce-product-recommendation",
        "financial": "financial-advisor",
        "healthcare": "healthcare-triage",
        "code-generation": "code-generation",
        "customer-support": "customer-support",
        "mcp-tools": "mcp-tool-server",
        "rag-agents": "rag-agent",
        "ecommerce-order": "ecommerce-order-management",
        "ecommerce-cs": "ecommerce-customer-service",
        "retail": "retail-shopping-assistant",
        "telecom": "telecom-customer-support",
        "research": "research-assistant",
    }

    for domain_dir in sorted(SCENARIOS_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue

        domain_name = domain_dir.name
        if domain_filter and domain_name != domain_filter:
            continue

        contract_name = domain_contract_map.get(domain_name)
        if not contract_name:
            continue

        contract_path = CONTRACTS_DIR / f"{contract_name}.yaml"
        if not contract_path.exists():
            print(f"  SKIP {domain_name}: contract {contract_name}.yaml not found")
            continue

        contract = aa.load(str(contract_path))

        summary = DomainSummary(domain=domain_name)
        results = run_domain(domain_dir, contract)

        for r in results:
            summary.total += 1
            report.total_scenarios += 1

            if r.passed:
                summary.passed += 1
                report.total_passed += 1
            else:
                summary.failed += 1
                report.total_failed += 1
                report.failures.append(r)

            # Track TP/FP/FN for hard violations
            if r.expected_hard > 0 and r.actual_hard > 0:
                summary.hard_tp += 1
            elif r.expected_hard == 0 and r.actual_hard > 0:
                summary.hard_fp += 1
            elif r.expected_hard > 0 and r.actual_hard == 0:
                summary.hard_fn += 1

            # Track TP/FP/FN for soft violations
            if r.expected_soft > 0 and r.actual_soft > 0:
                summary.soft_tp += 1
            elif r.expected_soft == 0 and r.actual_soft > 0:
                summary.soft_fp += 1
            elif r.expected_soft > 0 and r.actual_soft == 0:
                summary.soft_fn += 1

            if verbose and not r.passed:
                print(f"  FAIL {r.scenario_id}: {r.detail}")

        report.domains[domain_name] = summary

    return report


def print_report(report: BenchmarkReport) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 70)
    print("AgentContract-Bench v2 — Results")
    print("=" * 70)

    for domain_name, summary in sorted(report.domains.items()):
        pct = (summary.passed / summary.total * 100) if summary.total > 0 else 0
        status = "PASS" if summary.failed == 0 else "FAIL"

        # Compute precision/recall/F1 for hard violations
        h_denom_p = summary.hard_tp + summary.hard_fp
        h_precision = summary.hard_tp / h_denom_p if h_denom_p > 0 else 1.0
        h_denom_r = summary.hard_tp + summary.hard_fn
        h_recall = summary.hard_tp / h_denom_r if h_denom_r > 0 else 1.0
        h_f1 = (
            2 * h_precision * h_recall / (h_precision + h_recall)
            if (h_precision + h_recall) > 0
            else 0.0
        )

        # Compute precision/recall/F1 for soft violations
        s_denom_p = summary.soft_tp + summary.soft_fp
        s_precision = summary.soft_tp / s_denom_p if s_denom_p > 0 else 1.0
        s_denom_r = summary.soft_tp + summary.soft_fn
        s_recall = summary.soft_tp / s_denom_r if s_denom_r > 0 else 1.0
        s_f1 = (
            2 * s_precision * s_recall / (s_precision + s_recall)
            if (s_precision + s_recall) > 0
            else 0.0
        )

        print(
            f"  [{status}] {domain_name:20s} "
            f"{summary.passed}/{summary.total} ({pct:.0f}%)"
        )
        print(
            f"         hard P/R/F1={h_precision:.2f}/{h_recall:.2f}/{h_f1:.2f}"
            f"  soft P/R/F1={s_precision:.2f}/{s_recall:.2f}/{s_f1:.2f}"
        )

    print("-" * 70)
    total_pct = (
        report.total_passed / report.total_scenarios * 100
        if report.total_scenarios > 0
        else 0
    )
    print(
        f"  TOTAL: {report.total_passed}/{report.total_scenarios} "
        f"({total_pct:.0f}%)"
    )

    if report.failures:
        print(f"\n  {len(report.failures)} failures:")
        for f in report.failures[:10]:
            print(f"    - {f.scenario_id}: {f.detail}")
        if len(report.failures) > 10:
            print(f"    ... and {len(report.failures) - 10} more")

    print("=" * 70)


def report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    """Convert a BenchmarkReport to a JSON-serializable dict."""
    domains_out: dict[str, Any] = {}
    for domain_name, summary in sorted(report.domains.items()):
        h_denom_p = summary.hard_tp + summary.hard_fp
        h_precision = summary.hard_tp / h_denom_p if h_denom_p > 0 else 1.0
        h_denom_r = summary.hard_tp + summary.hard_fn
        h_recall = summary.hard_tp / h_denom_r if h_denom_r > 0 else 1.0
        h_f1 = (
            2 * h_precision * h_recall / (h_precision + h_recall)
            if (h_precision + h_recall) > 0
            else 0.0
        )
        s_denom_p = summary.soft_tp + summary.soft_fp
        s_precision = summary.soft_tp / s_denom_p if s_denom_p > 0 else 1.0
        s_denom_r = summary.soft_tp + summary.soft_fn
        s_recall = summary.soft_tp / s_denom_r if s_denom_r > 0 else 1.0
        s_f1 = (
            2 * s_precision * s_recall / (s_precision + s_recall)
            if (s_precision + s_recall) > 0
            else 0.0
        )
        domains_out[domain_name] = {
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "hard": {
                "tp": summary.hard_tp,
                "fp": summary.hard_fp,
                "fn": summary.hard_fn,
                "precision": round(h_precision, 4),
                "recall": round(h_recall, 4),
                "f1": round(h_f1, 4),
            },
            "soft": {
                "tp": summary.soft_tp,
                "fp": summary.soft_fp,
                "fn": summary.soft_fn,
                "precision": round(s_precision, 4),
                "recall": round(s_recall, 4),
                "f1": round(s_f1, 4),
            },
        }
    return {
        "total_scenarios": report.total_scenarios,
        "total_passed": report.total_passed,
        "total_failed": report.total_failed,
        "domains": domains_out,
        "failures": [asdict(f) for f in report.failures],
    }


def main() -> None:
    """CLI entry point."""
    domain_filter = None
    verbose = False
    output_format = "text"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--domain" and i + 1 < len(args):
            domain_filter = args[i + 1]
            i += 2
        elif args[i] == "--verbose":
            verbose = True
            i += 1
        elif args[i] == "--format" and i + 1 < len(args):
            output_format = args[i + 1]
            i += 2
        else:
            i += 1

    report = run_all(domain_filter=domain_filter, verbose=verbose)

    if output_format == "json":
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        print_report(report)

    sys.exit(0 if report.total_failed == 0 else 1)


if __name__ == "__main__":
    main()
