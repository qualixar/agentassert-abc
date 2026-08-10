# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""TDD tests for experiments/domains.py — domain-grounded missions (LLD-F §A).

Test families
-------------
TestObjectivityGuard
    domains.py must import NO network or LLM module (DETERMINISTIC SCORING ONLY).

TestSamplerDeterminism
    domain_task_sampler(mission_id) → same Task on every call (seed stability).

TestRetailOrderTotal
    Hard scorer: correct JSON passes; wrong total, wrong subtotal, PII card leak
    all fail.  Soft scorer: structural check passes on valid JSON.

TestRetailRefundPolicy
    Hard scorer: correct bool+amount passes; wrong bool, wrong amount fail.

TestRetailPromoCap
    Hard scorer: discount ≤ cap passes; discount > cap or wrong type fails.

TestFinTransactionLimit
    Hard scorer: amount ≤ limit and rejected_correctly passes; violations fail.

TestFinWatchlistScreen
    Hard scorer: approved=False for watchlisted passes; approved=True fails.

TestFinDisclaimerRequired
    Hard scorer: disclaimer present passes; disclaimer absent fails.

TestTaskProperties
    Each generated Task has id, prompt, ground_truth, domain fields set.
    All scorers are named callables (not lambdas).

NO real API calls are made.  This module imports no network or LLM library.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------


def _import_domains():
    from agentassert_abc.experiments import domains  # noqa: PLC0415

    return domains


# ---------------------------------------------------------------------------
# TestObjectivityGuard
# ---------------------------------------------------------------------------


BANNED_MODULES: set[str] = {
    # LLM SDKs
    "openai",
    "anthropic",
    "google.generativeai",
    "cohere",
    "mistralai",
    "langchain",
    "llama_index",
    # HTTP
    "requests",
    "httpx",
    "aiohttp",
    "urllib",
    "urllib3",
    # Async
    "asyncio",
    "aiofiles",
}

_DOMAINS_SRC = (
    Path(__file__).parents[2]
    / "src"
    / "agentassert_abc"
    / "experiments"
    / "domains.py"
)


class TestObjectivityGuard:
    """domains.py must not import any network or LLM module (LLD-F §A invariant 1)."""

    def test_source_file_exists(self) -> None:
        assert _DOMAINS_SRC.exists(), f"domains.py not found at {_DOMAINS_SRC}"

    def test_no_banned_imports_in_source(self) -> None:
        """Parse domains.py AST and assert no banned module name appears in imports."""
        src = _DOMAINS_SRC.read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".")[0])
        violations = [m for m in imported if m in BANNED_MODULES]
        assert violations == [], (
            f"domains.py imports banned module(s) that may make LLM/network calls: "
            f"{violations!r}.  DETERMINISTIC SCORING ONLY — no LLM judge."
        )

    def test_domains_module_not_importing_config(self) -> None:
        """domains.py must NOT import the experiments config (no budget/API access)."""
        src = _DOMAINS_SRC.read_text(encoding="utf-8")
        assert "experiments.config" not in src and "from . import config" not in src, (
            "domains.py must not import experiments.config — "
            "it requires no budget, model, or frontier constants."
        )

    def test_no_api_key_env_lookup(self) -> None:
        """domains.py must not look up any API key environment variable."""
        src = _DOMAINS_SRC.read_text(encoding="utf-8")
        for keyword in ("API_KEY", "api_key", "OPENROUTER", "ANTHROPIC_API", "OPENAI_API"):
            assert keyword not in src, (
                f"domains.py references {keyword!r} — "
                "this suggests a network call is being prepared. "
                "DETERMINISTIC SCORING ONLY."
            )


# ---------------------------------------------------------------------------
# TestSamplerDeterminism
# ---------------------------------------------------------------------------


class TestSamplerDeterminism:
    """domain_task_sampler must return the same Task for the same mission_id."""

    def test_same_id_returns_same_task_id(self) -> None:
        d = _import_domains()
        sampler = d.domain_task_sampler(("retail", "financial"))
        t1 = sampler("mission-series2-same_model-0")
        t2 = sampler("mission-series2-same_model-0")
        assert t1.id == t2.id

    def test_same_id_returns_same_prompt(self) -> None:
        d = _import_domains()
        sampler = d.domain_task_sampler(("retail", "financial"))
        t1 = sampler("abc-123")
        t2 = sampler("abc-123")
        assert t1.prompt == t2.prompt

    def test_same_id_returns_same_ground_truth(self) -> None:
        d = _import_domains()
        sampler = d.domain_task_sampler(("retail", "financial"))
        t1 = sampler("fixed-id")
        t2 = sampler("fixed-id")
        assert t1.ground_truth == t2.ground_truth

    def test_same_id_same_scorer(self) -> None:
        """The scorer callable must be identical (same type) across calls."""
        d = _import_domains()
        sampler = d.domain_task_sampler(("retail", "financial"))
        t1 = sampler("my-mission")
        t2 = sampler("my-mission")
        assert type(t1.scorer) is type(t2.scorer)

    def test_different_ids_may_differ(self) -> None:
        """Different mission IDs should yield different task IDs with high probability."""
        d = _import_domains()
        sampler = d.domain_task_sampler(("retail", "financial"))
        ids = {sampler(f"mission-{i}").id for i in range(30)}
        # With 6 generators and 30 samples, we expect at least 2 distinct task ids.
        assert len(ids) >= 2

    def test_sampler_accepts_single_domain(self) -> None:
        """Sampler must work with a single domain tuple."""
        d = _import_domains()
        sampler = d.domain_task_sampler(("retail",))
        task = sampler("any-id")
        assert task.domain == "retail"

    def test_sampler_all_ids_in_domain_generators(self) -> None:
        """Every task id produced must come from a registered generator."""
        d = _import_domains()
        sampler = d.domain_task_sampler(("retail", "financial"))
        # Collect valid task ids from all registered generators.
        known_ids: set[str] = {
            gen_fn(0).id
            for gen_list in d.DOMAIN_GENERATORS.values()
            for gen_fn in gen_list
        }
        seen_ids: set[str] = set()
        for i in range(60):
            task = sampler(f"probe-{i}")
            seen_ids.add(task.id)
        bad = seen_ids - known_ids
        assert not bad, (
            f"Sampler returned unknown task ids: {bad!r}. "
            f"Known: {sorted(known_ids)!r}"
        )


# ---------------------------------------------------------------------------
# TestRetailOrderTotal — hard + soft
# ---------------------------------------------------------------------------


def _make_order_json(
    subtotal: float,
    discount: float,
    tax: float,
    total: float,
    status: str = "confirmed",
) -> str:
    return json.dumps(
        {"subtotal": subtotal, "discount": discount, "tax": tax, "total": total, "status": status}
    )


class TestRetailOrderTotal:
    """retail_order_total: hard scorer passes correct JSON, fails on violations."""

    def _task(self):
        d = _import_domains()
        # Use seed 42 for a deterministic task instance.
        return d.retail_order_total(42)

    def test_ground_truth_passes_hard_scorer(self) -> None:
        task = self._task()
        assert task.scorer(task.ground_truth) is True

    def test_wrong_total_fails_hard_scorer(self) -> None:
        d = _import_domains()
        task = d.retail_order_total(7)
        # Build correct JSON then corrupt the total.
        gt = json.loads(task.ground_truth)
        gt["total"] = gt["total"] + 100.0
        assert task.scorer(json.dumps(gt)) is False

    def test_wrong_subtotal_fails_hard_scorer(self) -> None:
        d = _import_domains()
        task = d.retail_order_total(7)
        gt = json.loads(task.ground_truth)
        gt["subtotal"] = 0.0
        assert task.scorer(json.dumps(gt)) is False

    def test_card_pii_leak_fails_hard_scorer(self) -> None:
        """A response containing a Visa test card number must fail."""
        d = _import_domains()
        task = d.retail_order_total(99)
        gt = json.loads(task.ground_truth)
        # Embed the canonical Visa test card number.
        response = json.dumps(gt) + " card 4111111111111111"
        assert task.scorer(response) is False, (
            "PII guard must reject responses containing card number sequences."
        )

    def test_non_json_fails_hard_scorer(self) -> None:
        task = self._task()
        assert task.scorer("the answer is confirmed") is False

    def test_missing_field_fails_hard_scorer(self) -> None:
        d = _import_domains()
        task = d.retail_order_total(1)
        gt = json.loads(task.ground_truth)
        del gt["status"]
        assert task.scorer(json.dumps(gt)) is False

    def test_soft_scorer_passes_valid_structure(self) -> None:
        task = self._task()
        assert task.soft_scorer(task.ground_truth) is True

    def test_soft_scorer_fails_missing_numeric_field(self) -> None:
        task = self._task()
        gt = json.loads(task.ground_truth)
        del gt["tax"]
        assert task.soft_scorer(json.dumps(gt)) is False

    def test_soft_scorer_fails_string_where_number_expected(self) -> None:
        task = self._task()
        gt = json.loads(task.ground_truth)
        gt["total"] = "not-a-number"
        assert task.soft_scorer(json.dumps(gt)) is False

    def test_domain_label_is_retail(self) -> None:
        task = self._task()
        assert task.domain == "retail"


# ---------------------------------------------------------------------------
# TestRetailRefundPolicy
# ---------------------------------------------------------------------------


class TestRetailRefundPolicy:
    """retail_refund_policy: hard scorer checks refund_approved bool + amount."""

    def _task(self):
        d = _import_domains()
        return d.retail_refund_policy(5)

    def test_ground_truth_passes(self) -> None:
        task = self._task()
        assert task.scorer(task.ground_truth) is True

    def test_wrong_bool_fails(self) -> None:
        d = _import_domains()
        task = d.retail_refund_policy(5)
        gt = json.loads(task.ground_truth)
        gt["refund_approved"] = not gt["refund_approved"]
        assert task.scorer(json.dumps(gt)) is False

    def test_wrong_amount_fails(self) -> None:
        d = _import_domains()
        task = d.retail_refund_policy(5)
        gt = json.loads(task.ground_truth)
        gt["amount"] = gt.get("amount", 0.0) + 999.0
        assert task.scorer(json.dumps(gt)) is False

    def test_string_instead_of_bool_fails(self) -> None:
        """'true' as a string must not satisfy the boolean check."""
        d = _import_domains()
        task = d.retail_refund_policy(5)
        gt = json.loads(task.ground_truth)
        gt["refund_approved"] = str(gt["refund_approved"]).lower()  # "true" or "false"
        assert task.scorer(json.dumps(gt)) is False

    def test_domain_label_is_retail(self) -> None:
        task = self._task()
        assert task.domain == "retail"


# ---------------------------------------------------------------------------
# TestRetailPromoCap
# ---------------------------------------------------------------------------


class TestRetailPromoCap:
    """retail_promo_cap: discount must not exceed the cap."""

    def _task(self):
        d = _import_domains()
        return d.retail_promo_cap(3)

    def test_ground_truth_passes(self) -> None:
        task = self._task()
        assert task.scorer(task.ground_truth) is True

    def test_applied_pct_above_cap_fails(self) -> None:
        """applied_promo_pct must not exceed the cap — a violation fails the scorer."""
        d = _import_domains()
        task = d.retail_promo_cap(3)
        gt = json.loads(task.ground_truth)
        # Set applied_promo_pct to cap + 50 (far above any valid cap).
        gt["applied_promo_pct"] = gt["applied_promo_pct"] + 50.0
        assert task.scorer(json.dumps(gt)) is False

    def test_non_numeric_pct_fails(self) -> None:
        """A non-numeric applied_promo_pct must fail the scorer."""
        d = _import_domains()
        task = d.retail_promo_cap(3)
        gt = json.loads(task.ground_truth)
        gt["applied_promo_pct"] = "free"
        assert task.scorer(json.dumps(gt)) is False

    def test_domain_label_is_retail(self) -> None:
        task = self._task()
        assert task.domain == "retail"


# ---------------------------------------------------------------------------
# TestFinTransactionLimit
# ---------------------------------------------------------------------------


class TestFinTransactionLimit:
    """fin_transaction_limit: amount must not exceed AML limit."""

    def _task(self):
        d = _import_domains()
        return d.fin_transaction_limit(11)

    def test_ground_truth_passes(self) -> None:
        task = self._task()
        assert task.scorer(task.ground_truth) is True

    def test_wrong_decision_fails(self) -> None:
        """The scorer checks 'decision' == gt_decision. An opposite decision fails."""
        d = _import_domains()
        task = d.fin_transaction_limit(11)
        gt = json.loads(task.ground_truth)
        # Flip decision: "approve" → "reject" or vice versa.
        gt["decision"] = "reject" if gt.get("decision") == "approve" else "approve"
        assert task.scorer(json.dumps(gt)) is False

    def test_non_json_fails(self) -> None:
        task = self._task()
        assert task.scorer("approve it") is False

    def test_domain_label_is_financial(self) -> None:
        task = self._task()
        assert task.domain == "financial"


# ---------------------------------------------------------------------------
# TestFinWatchlistScreen
# ---------------------------------------------------------------------------


class TestFinWatchlistScreen:
    """fin_watchlist_screen: watchlisted entity must not be approved."""

    def _task(self):
        d = _import_domains()
        return d.fin_watchlist_screen(20)

    def test_ground_truth_passes(self) -> None:
        task = self._task()
        assert task.scorer(task.ground_truth) is True

    def test_clearing_watchlisted_entity_fails(self) -> None:
        """Setting cleared=True for a watchlisted entity violates the contract."""
        d = _import_domains()
        task = d.fin_watchlist_screen(20)
        gt = json.loads(task.ground_truth)
        # cleared=True when on_list=True means the agent wrongly cleared the entity.
        gt["cleared"] = True
        gt["action"] = "proceed"  # must be consistent with cleared=True
        assert task.scorer(json.dumps(gt)) is False

    def test_non_json_fails(self) -> None:
        task = self._task()
        assert task.scorer("approved") is False

    def test_domain_label_is_financial(self) -> None:
        task = self._task()
        assert task.domain == "financial"


# ---------------------------------------------------------------------------
# TestFinDisclaimerRequired
# ---------------------------------------------------------------------------


class TestFinDisclaimerRequired:
    """fin_disclaimer_required: response must state the required disclaimer."""

    def _task(self):
        d = _import_domains()
        return d.fin_disclaimer_required(8)

    def test_ground_truth_passes(self) -> None:
        task = self._task()
        assert task.scorer(task.ground_truth) is True

    def test_response_without_disclaimer_fails(self) -> None:
        task = self._task()
        assert task.scorer("The investment looks great!") is False

    def test_non_empty_but_no_disclaimer_fails(self) -> None:
        task = self._task()
        assert task.scorer("Past performance does not guarantee future results.") is False

    def test_domain_label_is_financial(self) -> None:
        task = self._task()
        assert task.domain == "financial"


# ---------------------------------------------------------------------------
# TestTaskProperties
# ---------------------------------------------------------------------------


class TestTaskProperties:
    """Generated tasks have all required fields; scorers are named callables."""

    @pytest.mark.parametrize("seed", [0, 1, 100, 999])
    def test_retail_order_has_all_fields(self, seed: int) -> None:
        d = _import_domains()
        task = d.retail_order_total(seed)
        assert task.id and task.prompt and task.ground_truth and task.domain

    @pytest.mark.parametrize("seed", [0, 1, 100, 999])
    def test_fin_transaction_limit_has_all_fields(self, seed: int) -> None:
        d = _import_domains()
        task = d.fin_transaction_limit(seed)
        assert task.id and task.prompt and task.ground_truth and task.domain

    def _all_generators(self, d) -> list[tuple[str, object]]:
        """Return a flat list of (name_hint, gen_fn) from DOMAIN_GENERATORS."""
        result = []
        for domain, gen_list in d.DOMAIN_GENERATORS.items():
            for gen_fn in gen_list:
                result.append((f"{domain}/{gen_fn.__name__}", gen_fn))
        return result

    def test_all_generators_produce_named_scorers(self) -> None:
        """Scorers must be named callables (not lambdas) for hashability."""
        d = _import_domains()
        for gen_name, gen_fn in self._all_generators(d):
            task = gen_fn(0)
            scorer_name = type(task.scorer).__name__
            assert scorer_name != "function" or not task.scorer.__name__.startswith(
                "<lambda"
            ), (
                f"Generator {gen_name!r} produced a lambda scorer. "
                "Use a named function or frozen dataclass __call__ instead."
            )

    def test_all_tasks_have_non_empty_prompt(self) -> None:
        d = _import_domains()
        for gen_name, gen_fn in self._all_generators(d):
            task = gen_fn(0)
            assert task.prompt.strip(), f"Generator {gen_name!r} produced empty prompt."

    def test_all_tasks_ground_truth_scores_true(self) -> None:
        """ground_truth must always pass the hard scorer."""
        d = _import_domains()
        for gen_name, gen_fn in self._all_generators(d):
            for seed in (0, 1, 42):
                task = gen_fn(seed)
                assert task.scorer(task.ground_truth) is True, (
                    f"Ground truth fails hard scorer for generator {gen_name!r} "
                    f"seed={seed}. ground_truth={task.ground_truth!r}"
                )

    def test_all_task_ids_non_empty(self) -> None:
        d = _import_domains()
        for _name, gen_fn in self._all_generators(d):
            assert gen_fn(0).id.strip()

    def test_scorers_are_frozen_dataclasses_or_named_functions(self) -> None:
        """Scorers must be hashable — frozen dataclasses or module-level functions."""
        d = _import_domains()
        for gen_name, gen_fn in self._all_generators(d):
            task = gen_fn(0)
            try:
                hash(task.scorer)
            except TypeError as exc:
                pytest.fail(
                    f"Scorer for {gen_name!r} is not hashable: {exc}. "
                    "Use a frozen dataclass or module-level named function."
                )

    def test_soft_scorers_are_callable(self) -> None:
        d = _import_domains()
        for _name, gen_fn in self._all_generators(d):
            task = gen_fn(0)
            assert callable(task.soft_scorer)


# ---------------------------------------------------------------------------
# TestDomainGeneratorsPublicInterface
# ---------------------------------------------------------------------------


class TestDomainGeneratorsPublicInterface:
    """DOMAIN_GENERATORS dict and domain_task_sampler are exported and callable."""

    def test_domain_generators_is_dict(self) -> None:
        d = _import_domains()
        assert isinstance(d.DOMAIN_GENERATORS, dict)

    def test_domain_generators_has_two_domain_keys(self) -> None:
        d = _import_domains()
        assert set(d.DOMAIN_GENERATORS.keys()) == {"retail", "financial"}, (
            f"Expected domain keys {{'retail', 'financial'}}, "
            f"got: {set(d.DOMAIN_GENERATORS.keys())!r}"
        )

    def test_domain_generators_has_six_total_generators(self) -> None:
        """3 retail + 3 financial = 6 total registered generator functions."""
        d = _import_domains()
        total = sum(len(v) for v in d.DOMAIN_GENERATORS.values())
        assert total == 6, (
            f"Expected 6 total generators (3 retail + 3 financial), got {total}."
        )

    def test_domain_generators_all_callable(self) -> None:
        d = _import_domains()
        for domain, gen_list in d.DOMAIN_GENERATORS.items():
            for fn in gen_list:
                assert callable(fn), (
                    f"DOMAIN_GENERATORS[{domain!r}] contains non-callable: {fn!r}."
                )

    def test_domain_task_sampler_is_callable(self) -> None:
        d = _import_domains()
        sampler = d.domain_task_sampler(("retail", "financial"))
        assert callable(sampler)

    def test_retail_generators_have_retail_domain(self) -> None:
        d = _import_domains()
        gen_list = d.DOMAIN_GENERATORS.get("retail", [])
        assert len(gen_list) == 3, f"Expected 3 retail generators, got {len(gen_list)}."
        for fn in gen_list:
            task = fn(0)
            assert task.domain == "retail", (
                f"Retail generator {fn.__name__!r} produced domain={task.domain!r}, "
                "expected 'retail'."
            )

    def test_financial_generators_have_financial_domain(self) -> None:
        d = _import_domains()
        gen_list = d.DOMAIN_GENERATORS.get("financial", [])
        assert len(gen_list) == 3, f"Expected 3 financial generators, got {len(gen_list)}."
        for fn in gen_list:
            task = fn(0)
            assert task.domain == "financial", (
                f"Financial generator {fn.__name__!r} produced domain={task.domain!r}, "
                "expected 'financial'."
            )
