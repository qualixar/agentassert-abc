# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Guard CLI: argument handling and start-up refusals."""

from __future__ import annotations

import pytest

from agentassert_abc.mcp import cli

from .conftest import CONTRACTS


class TestParser:
    def test_collects_the_downstream_command_after_the_separator(self) -> None:
        args = cli.build_parser().parse_args(["--contract", "c.yaml", "--", "npx", "-y", "srv"])
        assert args.command == ["--", "npx", "-y", "srv"]
        assert args.contract == "c.yaml"

    def test_downstream_flags_are_not_parsed_as_our_own(self) -> None:
        # The downstream server has its own flags; REMAINDER must hand them over
        # untouched rather than erroring on an unknown option.
        args = cli.build_parser().parse_args(
            ["--contract", "c.yaml", "--", "srv", "--fail-closed", "--contract", "theirs"]
        )
        assert args.command[1:] == ["srv", "--fail-closed", "--contract", "theirs"]
        assert args.fail_closed is False
        assert args.contract == "c.yaml"

    def test_defaults(self) -> None:
        args = cli.build_parser().parse_args(["--contract", "c.yaml", "--", "srv"])
        assert args.server_label == "mcp"
        assert args.session_id is None
        assert args.fail_closed is False

    def test_contract_is_required(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--", "srv"])


class TestStartupRefusals:
    def test_missing_downstream_command_exits_two(self, capsys) -> None:
        code = cli.main(["--contract", str(CONTRACTS / "safety-minimal.yaml")])
        assert code == 2
        assert "no downstream server command" in capsys.readouterr().err

    def test_bare_separator_with_no_command_exits_two(self, capsys) -> None:
        code = cli.main(["--contract", str(CONTRACTS / "safety-minimal.yaml"), "--"])
        assert code == 2
        assert "no downstream server command" in capsys.readouterr().err

    def test_unreadable_contract_exits_one_with_a_reason(self, capsys) -> None:
        code = cli.main(["--contract", "/nonexistent/c.yaml", "--", "srv"])
        assert code == 1
        assert "[agentassert]" in capsys.readouterr().err

    def test_unreadable_contract_file_exits_one(self, tmp_path, capsys) -> None:
        # A permission error is not a ContractLoadError, so it needs its own
        # handler — without one it would escape as a traceback.
        import os

        if os.geteuid() == 0:
            pytest.skip("root bypasses file permissions")
        contract = tmp_path / "locked.yaml"
        contract.write_text("name: x\n")
        contract.chmod(0o000)
        try:
            code = cli.main(["--contract", str(contract), "--", "srv"])
        finally:
            contract.chmod(0o644)
        assert code == 1
        assert "could not start guard" in capsys.readouterr().err

    def test_invalid_contract_exits_one(self, capsys) -> None:
        code = cli.main(["--contract", str(CONTRACTS / "invalid-missing-name.yaml"), "--", "srv"])
        assert code == 1
        assert "[agentassert]" in capsys.readouterr().err

    def test_contract_this_surface_cannot_evaluate_is_refused_loudly(
        self, tmp_path, capsys
    ) -> None:
        # A contract over state the guard never sees would score as a violation
        # on every call. The guard is launched by the client at start-up, so it
        # exits non-zero to land in the client's MCP server log rather than
        # running on as a silently disabled guard.
        contract = tmp_path / "unusable.yaml"
        contract.write_text(
            "dsl_version: '0.4'\n"
            "contractspec: '1.0'\n"
            "kind: agent\n"
            "name: unusable\n"
            "description: 'references state the guard never observes'\n"
            "version: '0.1'\n"
            "invariants:\n"
            "  hard:\n"
            "    - name: needs-db\n"
            "      description: 'nothing written to the database'\n"
            "      check:\n"
            "        field: database.rows_written\n"
            "        equals: 0\n"
        )
        code = cli.main(["--contract", str(contract), "--", "srv"])
        assert code == 1
        err = capsys.readouterr().err
        assert "database.rows_written" in err
        assert "MCP guard" in err

    def test_contract_over_the_response_surface_is_accepted(self, capsys) -> None:
        # Counterpart to the refusal above: `output.*` is derived from the tool
        # result, so full-governance.yaml must load. It gets past the gate and
        # fails later, on the downstream spawn.
        code = cli.main(
            [
                "--contract",
                str(CONTRACTS / "full-governance.yaml"),
                "--",
                "definitely-not-a-real-binary-xyz",
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "could not start downstream server" in err
        assert "cannot be enforced" not in err

    def test_missing_downstream_binary_does_not_leak_a_traceback(self, capsys) -> None:
        # stdout is the client's JSON-RPC stream. A traceback there would
        # desynchronise its parser on top of the underlying misconfiguration.
        code = cli.main(
            [
                "--contract",
                str(CONTRACTS / "safety-minimal.yaml"),
                "--",
                "definitely-not-a-real-binary-xyz",
            ]
        )
        captured = capsys.readouterr()
        assert code == 1
        assert captured.out == ""
        assert "Traceback" not in captured.err


class TestEntryPoint:
    def test_cli_raises_systemexit_with_the_return_code(self) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.cli()
        assert exc.value.code != 0
