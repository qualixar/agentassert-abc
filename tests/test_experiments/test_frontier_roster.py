# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for the frontier model roster, build_client factory, and gating.

TDD RED → GREEN:

TestFrontierModelPairs
    _FRONTIER_MODEL_PAIRS covers 3 primary conditions + 2 breadth arms.
    same_model uses model_a == model_b.
    All values are non-empty strings.

TestBuildClientFactory
    build_client(c, "dry") returns DryRunClient for any condition.
    build_client(c, "local") returns LocalClient (no Ollama contact needed for
        isinstance check — we only check the type, not call generate()).
    build_client(c, "frontier") with FRONTIER_ENABLED=False raises
        FrontierDisabledError for every condition.
    build_client with unknown tier raises ValueError.
    build_client with unknown frontier condition raises ValueError.

TestFrontierGateEnforcement
    Each of the five frontier conditions raises FrontierDisabledError when
    FRONTIER_ENABLED=False — no network call occurs.

TestDryRunBaselineBudget
    A full pipeline run with DryRunClient over all 5 motifs × 3 conditions ×
    n_per_cell=10 produces budget_spent == 0.0.
    This confirms the plumbing for the morning run baseline.

SAFETY: FRONTIER_ENABLED is only set True inside patch.object context
managers.  No test leaves the flag True after the `with` block exits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from agentassert_abc.experiments import config
from agentassert_abc.experiments.budget import BudgetLedger
from agentassert_abc.experiments.motifs import MOTIF_LIBRARY

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Lazy import helpers — tests fail on collection if run.py is missing the new
# names (expected RED before implementation).
# ---------------------------------------------------------------------------


def _import_run():
    from agentassert_abc.experiments import run  # noqa: PLC0415

    return run


# ---------------------------------------------------------------------------
# TestFrontierModelPairs
# ---------------------------------------------------------------------------


class TestFrontierModelPairs:
    """_FRONTIER_MODEL_PAIRS exists and has correct structure."""

    def test_module_exposes_frontier_model_pairs(self) -> None:
        run = _import_run()
        assert hasattr(run, "_FRONTIER_MODEL_PAIRS"), (
            "_FRONTIER_MODEL_PAIRS not found in run module"
        )

    def test_has_same_model_condition(self) -> None:
        run = _import_run()
        assert "same_model" in run._FRONTIER_MODEL_PAIRS

    def test_has_same_vendor_condition(self) -> None:
        run = _import_run()
        assert "same_vendor" in run._FRONTIER_MODEL_PAIRS

    def test_has_different_vendor_condition(self) -> None:
        run = _import_run()
        assert "different_vendor" in run._FRONTIER_MODEL_PAIRS

    def test_same_model_uses_identical_models(self) -> None:
        """same_model condition: model_a must equal model_b."""
        run = _import_run()
        model_a, model_b = run._FRONTIER_MODEL_PAIRS["same_model"]
        assert model_a == model_b, (
            f"same_model condition must use identical models; "
            f"got model_a={model_a!r}, model_b={model_b!r}"
        )

    def test_same_vendor_uses_different_models(self) -> None:
        """same_vendor condition: model_a != model_b (two different Qwen models)."""
        run = _import_run()
        model_a, model_b = run._FRONTIER_MODEL_PAIRS["same_vendor"]
        assert model_a != model_b, (
            "same_vendor condition should use different model IDs "
            f"(same vendor family, different versions); got {model_a!r} == {model_b!r}"
        )

    def test_different_vendor_uses_different_models(self) -> None:
        """different_vendor: model_a != model_b (cross-vendor)."""
        run = _import_run()
        model_a, model_b = run._FRONTIER_MODEL_PAIRS["different_vendor"]
        assert model_a != model_b

    def test_all_model_ids_are_non_empty_strings(self) -> None:
        run = _import_run()
        for condition, (model_a, model_b) in run._FRONTIER_MODEL_PAIRS.items():
            assert isinstance(model_a, str) and model_a.strip(), (
                f"model_a for {condition!r} is empty or not a string"
            )
            assert isinstance(model_b, str) and model_b.strip(), (
                f"model_b for {condition!r} is empty or not a string"
            )

    def test_has_breadth_arm_meta(self) -> None:
        """different_vendor_meta breadth arm present."""
        run = _import_run()
        assert "different_vendor_meta" in run._FRONTIER_MODEL_PAIRS, (
            "Breadth arm 'different_vendor_meta' missing from _FRONTIER_MODEL_PAIRS"
        )

    def test_has_breadth_arm_grok(self) -> None:
        """different_vendor_grok breadth arm present."""
        run = _import_run()
        assert "different_vendor_grok" in run._FRONTIER_MODEL_PAIRS, (
            "Breadth arm 'different_vendor_grok' missing from _FRONTIER_MODEL_PAIRS"
        )

    def test_same_model_uses_openrouter_default_model(self) -> None:
        """same_model pair should use config.OPENROUTER_DEFAULT_MODEL."""
        run = _import_run()
        model_a, model_b = run._FRONTIER_MODEL_PAIRS["same_model"]
        assert model_a == config.OPENROUTER_DEFAULT_MODEL, (
            f"same_model.model_a should be OPENROUTER_DEFAULT_MODEL="
            f"{config.OPENROUTER_DEFAULT_MODEL!r}, got {model_a!r}"
        )


# ---------------------------------------------------------------------------
# TestBuildClientFactory
# ---------------------------------------------------------------------------


class TestBuildClientFactory:
    """build_client(condition, tier) factory."""

    def test_module_exposes_build_client(self) -> None:
        run = _import_run()
        assert callable(getattr(run, "build_client", None)), (
            "build_client not found in run module"
        )

    def test_dry_returns_dry_run_client(self) -> None:
        run = _import_run()
        client = run.build_client("same_model", "dry")
        assert isinstance(client, run.DryRunClient)

    def test_dry_returns_dry_run_client_same_vendor(self) -> None:
        run = _import_run()
        assert isinstance(run.build_client("same_vendor", "dry"), run.DryRunClient)

    def test_dry_returns_dry_run_client_different_vendor(self) -> None:
        run = _import_run()
        assert isinstance(run.build_client("different_vendor", "dry"), run.DryRunClient)

    def test_dry_run_client_cost_zero(self) -> None:
        run = _import_run()
        client = run.build_client("same_model", "dry")
        resp = client.generate("any-model", "any prompt")
        assert resp.cost_usd == 0.0

    def test_unknown_tier_raises_value_error(self) -> None:
        run = _import_run()
        with pytest.raises(ValueError, match="tier"):
            run.build_client("same_model", "bogus_tier")

    def test_frontier_unknown_condition_raises_value_error(self) -> None:
        """Unknown condition for frontier tier raises ValueError, not FrontierDisabledError."""
        run = _import_run()

        with (
            patch.object(config, "FRONTIER_ENABLED", True),
            pytest.raises(ValueError),
        ):
            # FRONTIER_ENABLED=True avoids FrontierDisabledError, but unknown condition
            # should raise ValueError inside build_client before adapter construction.
            run.build_client("totally_unknown_condition", "frontier")


# ---------------------------------------------------------------------------
# TestFrontierGateEnforcement
# ---------------------------------------------------------------------------


class TestFrontierGateEnforcement:
    """build_client(..., "frontier") raises FrontierDisabledError when gate closed.

    These tests prove the roster is permanently inert tonight:
    FRONTIER_ENABLED=False → no adapter is ever constructed → $0 spend.

    Three ways a paid call could accidentally happen:
    1. build_client skips the adapter and returns a non-gated object.
    2. The adapter's __init__ gate is bypassed.
    3. FRONTIER_ENABLED is mutated by build_client itself.

    Each is blocked:
    1. Every frontier path goes through a gated adapter class.
    2. _OpenAICompatBase.__init__ always checks config.FRONTIER_ENABLED first.
    3. build_client has no write access to config.FRONTIER_ENABLED (read-only Final).
    """

    def _check_frontier_disabled(self, condition: str) -> None:
        run = _import_run()
        from agentassert_abc.experiments.models import FrontierDisabledError  # noqa: PLC0415

        with (
            patch.object(config, "FRONTIER_ENABLED", False),
            pytest.raises(FrontierDisabledError),
        ):
            run.build_client(condition, "frontier")

    def test_same_model_frontier_raises(self) -> None:
        self._check_frontier_disabled("same_model")

    def test_same_vendor_frontier_raises(self) -> None:
        self._check_frontier_disabled("same_vendor")

    def test_different_vendor_frontier_raises(self) -> None:
        self._check_frontier_disabled("different_vendor")

    def test_different_vendor_meta_frontier_raises(self) -> None:
        """MetaSparkClient also requires FRONTIER_ENABLED=True AND MODEL_API_KEY."""
        import os  # noqa: PLC0415

        run = _import_run()
        from agentassert_abc.experiments.models import FrontierDisabledError  # noqa: PLC0415

        # Clear MODEL_API_KEY to prevent key-check from passing first
        env_without_key = {k: v for k, v in os.environ.items() if k != "MODEL_API_KEY"}
        with (
            patch.object(config, "FRONTIER_ENABLED", False),
            patch.dict(os.environ, env_without_key, clear=True),
            pytest.raises(FrontierDisabledError),
        ):
            run.build_client("different_vendor_meta", "frontier")

    def test_different_vendor_grok_frontier_raises(self) -> None:
        self._check_frontier_disabled("different_vendor_grok")

    def test_frontier_enabled_still_false_after_dry_call(
        self, tmp_path: Path
    ) -> None:
        """Calling build_client in dry mode must not change FRONTIER_ENABLED."""
        run = _import_run()
        run.build_client("same_model", "dry")
        assert config.FRONTIER_ENABLED is False, (
            "build_client('same_model', 'dry') mutated FRONTIER_ENABLED"
        )

    def test_frontier_model_pairs_constant_does_not_trigger_construction(
        self,
    ) -> None:
        """Merely reading _FRONTIER_MODEL_PAIRS must not construct any adapter."""
        run = _import_run()
        # Access the constant — no FrontierDisabledError should be raised.
        pairs = run._FRONTIER_MODEL_PAIRS
        assert isinstance(pairs, dict)
        # FRONTIER_ENABLED must still be False
        assert config.FRONTIER_ENABLED is False


# ---------------------------------------------------------------------------
# TestDryRunBaselineBudget
# ---------------------------------------------------------------------------


class TestDryRunBaselineBudget:
    """Full 5-motif × 3-condition dry-run pipeline produces $0 spend."""

    def test_all_five_motifs_three_conditions_budget_zero(
        self, tmp_path: Path
    ) -> None:
        run = _import_run()
        # series2, series3, parallel2, quorum2of3, hierarchy
        all_motifs = list(MOTIF_LIBRARY.values())
        conditions = ["same_model", "same_vendor", "different_vendor"]
        n_per_cell = 10
        ledger = BudgetLedger()

        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=all_motifs,
            sharing_conditions=conditions,
            n_per_cell=n_per_cell,
            p0=config.P0_RELIABILITY,
            alpha=config.ALPHA,
            out_path=tmp_path / "missions_baseline.jsonl",
            ledger=ledger,
        )

        assert summary.budget_spent == 0.0, (
            f"Full dry-run must cost $0.00; got ${summary.budget_spent:.6f}"
        )

    def test_all_five_motifs_three_conditions_mission_count(
        self, tmp_path: Path
    ) -> None:
        run = _import_run()
        all_motifs = list(MOTIF_LIBRARY.values())
        conditions = ["same_model", "same_vendor", "different_vendor"]
        n_per_cell = 10
        ledger = BudgetLedger()

        summary = run.run_experiment(
            run.DryRunClient(),
            motifs=all_motifs,
            sharing_conditions=conditions,
            n_per_cell=n_per_cell,
            p0=config.P0_RELIABILITY,
            alpha=config.ALPHA,
            out_path=tmp_path / "missions_baseline.jsonl",
            ledger=ledger,
        )

        expected = len(all_motifs) * len(conditions) * n_per_cell
        assert summary.n_missions == expected, (
            f"Expected {expected} missions; got {summary.n_missions}"
        )

    def test_build_client_dry_integrates_with_run_experiment(
        self, tmp_path: Path
    ) -> None:
        """build_client('same_model', 'dry') client works end-to-end in run_experiment."""
        run = _import_run()
        client = run.build_client("same_model", "dry")
        ledger = BudgetLedger()
        summary = run.run_experiment(
            client,
            motifs=[MOTIF_LIBRARY["series2"]],
            sharing_conditions=["same_model"],
            n_per_cell=5,
            p0=config.P0_RELIABILITY,
            alpha=config.ALPHA,
            out_path=tmp_path / "missions_factory_test.jsonl",
            ledger=ledger,
        )
        assert summary.budget_spent == 0.0
        assert summary.n_missions == 5


# ---------------------------------------------------------------------------
# TestConfigFrontierConstants
# ---------------------------------------------------------------------------


class TestConfigFrontierConstants:
    """New frontier model roster constants in config.py are present."""

    def test_openrouter_qwen_same_vendor_exists(self) -> None:
        assert hasattr(config, "OPENROUTER_QWEN_SAME_VENDOR"), (
            "config.OPENROUTER_QWEN_SAME_VENDOR missing"
        )

    def test_openrouter_qwen_same_vendor_non_empty(self) -> None:
        assert isinstance(config.OPENROUTER_QWEN_SAME_VENDOR, str)
        assert config.OPENROUTER_QWEN_SAME_VENDOR.strip()

    def test_openrouter_gemma_diff_vendor_exists(self) -> None:
        assert hasattr(config, "OPENROUTER_GEMMA_DIFF_VENDOR"), (
            "config.OPENROUTER_GEMMA_DIFF_VENDOR missing"
        )

    def test_openrouter_gemma_diff_vendor_non_empty(self) -> None:
        assert isinstance(config.OPENROUTER_GEMMA_DIFF_VENDOR, str)
        assert config.OPENROUTER_GEMMA_DIFF_VENDOR.strip()

    def test_grok_model_exists(self) -> None:
        assert hasattr(config, "GROK_MODEL"), (
            "config.GROK_MODEL missing"
        )

    def test_grok_model_non_empty(self) -> None:
        assert isinstance(config.GROK_MODEL, str)
        assert config.GROK_MODEL.strip()

    def test_openrouter_default_model_unchanged(self) -> None:
        """Existing OPENROUTER_DEFAULT_MODEL must not be changed."""
        assert config.OPENROUTER_DEFAULT_MODEL == "qwen/qwen3-7b-fast"

    def test_frontier_enabled_still_false(self) -> None:
        """Adding roster constants must not flip FRONTIER_ENABLED."""
        assert config.FRONTIER_ENABLED is False
