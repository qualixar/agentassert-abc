# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Hook installer and its CLI.

Every test here redirects ``Path.home()`` to a tmp dir. The installer writes to
and *unlinks from* ``~/.claude-hooks``, so an unpatched run would delete a real
user's installed hook.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from agentassert_abc.claude_code import cli as cli_mod
from agentassert_abc.claude_code.installer import _HOOK_FILENAME, install, uninstall

from ..test_mcp.conftest import CONTRACTS


@pytest.fixture(autouse=True)
def _fake_home(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("AGENTASSERT_CONTRACT", raising=False)
    return tmp_path


def hook_path(home: Path) -> Path:
    return home / ".claude-hooks" / _HOOK_FILENAME


class TestInstall:
    def test_creates_the_hooks_directory_and_copies_the_hook(self, tmp_path, capsys) -> None:
        install("/tmp/contract.yaml")
        installed = hook_path(tmp_path)
        assert installed.exists()
        assert "def main(" in installed.read_text()
        assert "installed" in capsys.readouterr().out

    def test_installed_hook_is_executable(self, tmp_path) -> None:
        # Claude Code executes the file directly; without the bit it silently
        # never runs.
        install("/tmp/contract.yaml")
        assert hook_path(tmp_path).stat().st_mode & 0o111

    def test_prints_the_env_var_the_user_must_export(self, capsys) -> None:
        install("/tmp/my-contract.yaml")
        out = capsys.readouterr().out
        assert "AGENTASSERT_CONTRACT=/tmp/my-contract.yaml" in out

    def test_refuses_to_clobber_an_existing_hook(self, tmp_path) -> None:
        install("/tmp/contract.yaml")
        hook_path(tmp_path).write_text("# hand-edited by the user\n")
        with pytest.raises(FileExistsError, match="--force"):
            install("/tmp/contract.yaml")
        assert "hand-edited" in hook_path(tmp_path).read_text()

    def test_force_overwrites(self, tmp_path) -> None:
        install("/tmp/contract.yaml")
        hook_path(tmp_path).write_text("# stale\n")
        install("/tmp/contract.yaml", force=True)
        assert "def main(" in hook_path(tmp_path).read_text()

    def test_is_idempotent_under_force(self, tmp_path) -> None:
        install("/tmp/contract.yaml")
        first = hook_path(tmp_path).read_text()
        install("/tmp/contract.yaml", force=True)
        assert hook_path(tmp_path).read_text() == first


class TestUninstall:
    def test_removes_an_installed_hook(self, tmp_path, capsys) -> None:
        install("/tmp/contract.yaml")
        uninstall()
        assert not hook_path(tmp_path).exists()
        assert "removed" in capsys.readouterr().out

    def test_is_a_no_op_when_nothing_is_installed(self, tmp_path, capsys) -> None:
        uninstall()
        assert "not installed" in capsys.readouterr().out

    def test_only_touches_its_own_file(self, tmp_path) -> None:
        # A user's other hooks live in the same directory.
        install("/tmp/contract.yaml")
        neighbour = tmp_path / ".claude-hooks" / "01-someone-elses-hook.py"
        neighbour.write_text("# not ours\n")
        uninstall()
        assert neighbour.exists()


class TestCli:
    def test_install_requires_an_existing_contract(self, tmp_path) -> None:
        result = CliRunner().invoke(cli_mod.cli, ["install", "-c", "/nonexistent/c.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.output
        assert not hook_path(tmp_path).exists()

    def test_install_writes_the_hook(self, tmp_path) -> None:
        result = CliRunner().invoke(
            cli_mod.cli, ["install", "-c", str(CONTRACTS / "safety-minimal.yaml")]
        )
        assert result.exit_code == 0, result.output
        assert hook_path(tmp_path).exists()

    def test_install_force_flag_is_wired(self, tmp_path) -> None:
        contract = str(CONTRACTS / "safety-minimal.yaml")
        CliRunner().invoke(cli_mod.cli, ["install", "-c", contract])
        clobbered = CliRunner().invoke(cli_mod.cli, ["install", "-c", contract])
        assert clobbered.exit_code != 0
        forced = CliRunner().invoke(cli_mod.cli, ["install", "-c", contract, "--force"])
        assert forced.exit_code == 0, forced.output

    def test_install_records_the_resolved_absolute_path(self, tmp_path) -> None:
        # A relative path in the exported env var would break the moment the
        # user starts Claude Code from another directory.
        result = CliRunner().invoke(
            cli_mod.cli,
            ["install", "-c", "tests/test_gateway/fixtures/contracts/safety-minimal.yaml"],
        )
        assert result.exit_code == 0, result.output
        assert "AGENTASSERT_CONTRACT=/" in result.output

    def test_uninstall_removes_it(self, tmp_path) -> None:
        CliRunner().invoke(cli_mod.cli, ["install", "-c", str(CONTRACTS / "safety-minimal.yaml")])
        result = CliRunner().invoke(cli_mod.cli, ["uninstall"])
        assert result.exit_code == 0
        assert not hook_path(tmp_path).exists()

    def test_status_reports_not_installed(self) -> None:
        result = CliRunner().invoke(cli_mod.cli, ["status"])
        assert "Hook not installed" in result.output
        assert "not set" in result.output

    def test_status_reports_installed_and_the_contract(self, monkeypatch) -> None:
        CliRunner().invoke(cli_mod.cli, ["install", "-c", str(CONTRACTS / "safety-minimal.yaml")])
        monkeypatch.setenv("AGENTASSERT_CONTRACT", "/tmp/c.yaml")
        result = CliRunner().invoke(cli_mod.cli, ["status"])
        assert "Hook installed" in result.output
        assert "/tmp/c.yaml" in result.output
