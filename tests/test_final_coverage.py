# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""Final coverage tests — close ALL remaining gaps."""

import pytest


class TestInitLazyImports:
    """__init__.py lazy import paths."""

    def test_parse(self) -> None:
        import agentassert_abc as aa

        assert callable(aa.parse)

    def test_validate(self) -> None:
        import agentassert_abc as aa

        assert callable(aa.validate)


class TestSPRTValidation:
    """sprt.py input validation paths."""

    def test_invalid_p0(self) -> None:
        from agentassert_abc.certification.sprt import SPRTCertifier

        with pytest.raises(ValueError, match="p0"):
            SPRTCertifier(p0=0.0, p1=0.95, alpha=0.05, beta=0.1)

    def test_invalid_p1(self) -> None:
        from agentassert_abc.certification.sprt import SPRTCertifier

        with pytest.raises(ValueError, match="p1"):
            SPRTCertifier(p0=0.85, p1=1.0, alpha=0.05, beta=0.1)

    def test_p0_gte_p1(self) -> None:
        from agentassert_abc.certification.sprt import SPRTCertifier

        with pytest.raises(ValueError, match="strictly less"):
            SPRTCertifier(p0=0.95, p1=0.90, alpha=0.05, beta=0.1)


class TestGenericAdapterEdges:
    """generic.py extract_state() paths and Protocol compliance."""

    def test_generic_adapter_satisfies_protocol(self) -> None:
        """H-06: GenericAdapter should satisfy the redesigned AgentAdapter protocol."""
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.integrations.base import AgentAdapter
        from agentassert_abc.integrations.generic import GenericAdapter

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
""")
        adapter = GenericAdapter(contract)
        assert isinstance(adapter, AgentAdapter)

    def test_extract_state_non_dict_raises(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.integrations.generic import GenericAdapter

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
""")
        adapter = GenericAdapter(contract)
        with pytest.raises(TypeError, match="expects dict"):
            adapter.extract_state("not a dict")

    def test_extract_state_dict(self) -> None:
        from agentassert_abc.dsl.parser import loads_contract
        from agentassert_abc.integrations.generic import GenericAdapter

        contract = loads_contract("""
contractspec: "0.1"
kind: agent
name: test
description: test
version: "1.0.0"
""")
        adapter = GenericAdapter(contract)
        state = adapter.extract_state({"key": "value"})
        assert state == {"key": "value"}
