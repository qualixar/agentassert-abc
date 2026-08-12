# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Tests for package version and basic imports."""


def test_runtime_version_matches_the_packaging_metadata() -> None:
    """`__version__` and pyproject must not drift apart.

    Previously this asserted a hardcoded literal, which had to be edited on
    every release and so only ever proved that someone had edited it. Comparing
    the two sources catches the failure that actually bites: a wheel published
    with one version while the runtime reports another.
    """
    import pathlib
    import re

    from agentassert_abc import __version__

    pyproject = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
    declared = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.MULTILINE)
    assert declared is not None, "pyproject.toml has no top-level version"
    assert __version__ == declared.group(1)


def test_version_is_pep440_release() -> None:
    import re

    from agentassert_abc import __version__

    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), __version__


def test_version_is_string() -> None:
    from agentassert_abc import __version__

    assert isinstance(__version__, str)
