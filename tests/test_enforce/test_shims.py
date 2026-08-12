# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""Framework shims, driven with fakes shaped like each framework's real context.

The frameworks themselves are deliberately not installed: the shims are
structurally typed, so what needs proving is the *translation* — native context
in, native veto convention out — not that a third-party package works.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentassert_abc.enforce import EnforcementBridge
from agentassert_abc.enforce.shims import (
    ContractFunctionMiddleware,
    crewai_after_tool_hook,
    crewai_before_tool_hook,
    langchain_tool_middleware,
    register_agentscope_hooks,
)
from agentassert_abc.enforce.shims.agentscope import (
    agentscope_post_acting_hook,
    agentscope_pre_acting_hook,
)
from agentassert_abc.exceptions import ContractBreachError

from ..test_mcp.conftest import StubEnforcer, allow, deny, modify


def bridge(decisions=None, **kw) -> EnforcementBridge:
    return EnforcementBridge(StubEnforcer(decisions), **kw)


class TestNoFrameworkImportsAtImportTime:
    def test_shims_import_without_any_agent_framework_installed(self) -> None:
        # The whole point of structural typing: installing AgentAssert must not
        # drag in CrewAI, LangChain, MAF or AgentScope.
        import importlib
        import sys

        for name in ("crewai", "langchain", "langchain_core", "agent_framework", "agentscope"):
            assert name not in sys.modules, f"{name} leaked into the test process"
        for mod in (
            "agentassert_abc.enforce.shims.crewai",
            "agentassert_abc.enforce.shims.langchain",
            "agentassert_abc.enforce.shims.maf",
            "agentassert_abc.enforce.shims.agentscope",
        ):
            importlib.import_module(mod)


# ---------------------------------------------------------------------------
# CrewAI — BeforeToolCallHook returns False to block
# ---------------------------------------------------------------------------


@dataclass
class CrewContext:
    """Shaped like crewai.hooks.types.ToolCallHookContext."""

    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_result: Any = None
    blocked_reason: str = ""


class TestCrewAI:
    def test_allowed_call_returns_true(self) -> None:
        hook = crewai_before_tool_hook(bridge([allow()]))
        assert hook(CrewContext("read", {"path": "a"})) is True

    def test_denied_call_returns_false(self) -> None:
        # False is CrewAI's block signal — the tool is not executed.
        hook = crewai_before_tool_hook(bridge([deny(reason="destructive")]))
        assert hook(CrewContext("rm")) is False

    def test_denied_call_records_the_reason(self) -> None:
        ctx = CrewContext("rm")
        crewai_before_tool_hook(bridge([deny(reason="destructive", violation="bl")]))(ctx)
        assert "destructive" in ctx.blocked_reason
        assert "bl" in ctx.blocked_reason

    def test_modify_rewrites_tool_input_in_place(self) -> None:
        # CrewAI reads tool_input after the hook chain, so the rewrite must land
        # on the context for MODIFY to have any effect.
        ctx = CrewContext("read", {"path": "/etc/passwd"})
        crewai_before_tool_hook(bridge([modify({"path": "/tmp/safe"})]))(ctx)
        assert ctx.tool_input == {"path": "/tmp/safe"}

    def test_allow_leaves_tool_input_untouched(self) -> None:
        ctx = CrewContext("read", {"path": "a"})
        crewai_before_tool_hook(bridge([allow()]))(ctx)
        assert ctx.tool_input == {"path": "a"}

    def test_after_hook_withholds_a_denied_result(self) -> None:
        ctx = CrewContext("read", {}, tool_result="sk-secret")
        assert crewai_after_tool_hook(bridge([deny(reason="leaked")]))(ctx) is False
        assert "sk-secret" not in ctx.tool_result
        assert "withheld" in ctx.tool_result
        assert "executed" in ctx.tool_result

    def test_after_hook_passes_a_clean_result(self) -> None:
        ctx = CrewContext("read", {}, tool_result="fine")
        assert crewai_after_tool_hook(bridge([allow()]))(ctx) is True
        assert ctx.tool_result == "fine"

    def test_missing_fields_do_not_crash_the_run(self) -> None:
        class Bare:
            pass

        assert crewai_before_tool_hook(bridge([allow()]))(Bare()) is True


# ---------------------------------------------------------------------------
# LangChain / LangGraph — wrap_tool_call short-circuits by not calling handler
# ---------------------------------------------------------------------------


@dataclass
class LcRequest:
    """Shaped like LangChain's ToolCallRequest."""

    tool_call: dict[str, Any]


def fake_message(content: str, call_id: str) -> dict[str, Any]:
    return {"content": content, "tool_call_id": call_id}


class TestLangChain:
    def test_allowed_call_invokes_the_handler(self) -> None:
        calls: list[Any] = []
        wrap = langchain_tool_middleware(bridge([allow(), allow()]), message_factory=fake_message)
        out = wrap(
            LcRequest({"name": "read", "args": {"p": 1}, "id": "c1"}),
            lambda r: calls.append(r) or "RESULT",
        )
        assert len(calls) == 1
        assert out == "RESULT"

    def test_denied_call_never_invokes_the_handler(self) -> None:
        # Not calling the handler is exactly how LangChain middleware blocks.
        calls: list[Any] = []
        wrap = langchain_tool_middleware(
            bridge([deny(reason="destructive")]), message_factory=fake_message
        )
        out = wrap(
            LcRequest({"name": "rm", "args": {}, "id": "c1"}),
            lambda r: calls.append(r) or "RESULT",
        )
        assert calls == [], "a denied tool must never execute"
        assert "destructive" in out["content"]
        assert out["tool_call_id"] == "c1"

    def test_modify_rewrites_the_request_before_the_handler_sees_it(self) -> None:
        seen: list[Any] = []
        wrap = langchain_tool_middleware(
            bridge([modify({"p": "safe"}), allow()]), message_factory=fake_message
        )
        wrap(
            LcRequest({"name": "read", "args": {"p": "danger"}, "id": "c1"}),
            lambda r: seen.append(r) or "R",
        )
        assert seen[0].tool_call["args"] == {"p": "safe"}

    def test_denied_result_is_withheld_after_execution(self) -> None:
        wrap = langchain_tool_middleware(
            bridge([allow(), deny(reason="leaked")]), message_factory=fake_message
        )
        out = wrap(LcRequest({"name": "read", "args": {}, "id": "c1"}), lambda _r: "sk-secret")
        assert "sk-secret" not in str(out)
        assert "withheld" in out["content"]

    def test_unevaluated_call_skips_result_scoring(self) -> None:
        # Fail-open on the pre-hook must not be followed by scoring the result,
        # which would blame the agent for our evaluation failure.
        enforcer = StubEnforcer(raises=RuntimeError("boom"))
        wrap = langchain_tool_middleware(EnforcementBridge(enforcer), message_factory=fake_message)
        assert wrap(LcRequest({"name": "read", "args": {}, "id": "c"}), lambda _r: "R") == "R"

    def test_reads_a_request_that_exposes_fields_directly(self) -> None:
        @dataclass
        class Flat:
            name: str
            args: dict[str, Any]
            id: str

        wrap = langchain_tool_middleware(bridge([deny()]), message_factory=fake_message)
        out = wrap(Flat("rm", {}, "c9"), lambda _r: "R")
        assert out["tool_call_id"] == "c9"

    def test_result_text_is_read_from_a_message_like_object(self) -> None:
        @dataclass
        class Msg:
            content: str

        enforcer = StubEnforcer([allow(), allow()])
        wrap = langchain_tool_middleware(EnforcementBridge(enforcer), message_factory=fake_message)
        wrap(LcRequest({"name": "read", "args": {}, "id": "c"}), lambda _r: Msg("hello"))
        post = enforcer.events[-1]
        assert post.state["output.text"] == "hello"

    def test_default_factory_requires_langchain(self) -> None:
        # Documents the one place LangChain is genuinely needed, and proves the
        # import is lazy: constructing the middleware must not raise.
        wrap = langchain_tool_middleware(bridge([deny()]))
        with pytest.raises(ImportError):
            wrap(LcRequest({"name": "rm", "args": {}, "id": "c"}), lambda _r: "R")


# ---------------------------------------------------------------------------
# Microsoft Agent Framework — decline to call next()
# ---------------------------------------------------------------------------


@dataclass
class MafFunction:
    name: str


@dataclass
class MafContext:
    """Shaped like agent_framework.FunctionInvocationContext."""

    function: MafFunction
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    terminate: bool = False


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestMicrosoftAgentFramework:
    def test_allowed_call_awaits_next(self) -> None:
        called: list[Any] = []

        async def nxt(ctx: Any) -> None:
            called.append(ctx)
            ctx.result = "RESULT"

        ctx = MafContext(MafFunction("read"), {"p": 1})
        run(ContractFunctionMiddleware(bridge([allow(), allow()])).process(ctx, nxt))
        assert len(called) == 1
        assert ctx.result == "RESULT"

    def test_denied_call_never_awaits_next(self) -> None:
        called: list[Any] = []

        async def nxt(ctx: Any) -> None:
            called.append(ctx)

        ctx = MafContext(MafFunction("rm"))
        run(ContractFunctionMiddleware(bridge([deny(reason="destructive")])).process(ctx, nxt))
        assert called == [], "a denied function must never execute"
        assert "destructive" in ctx.result
        assert ctx.terminate is True

    def test_modify_rewrites_arguments_before_next(self) -> None:
        seen: list[dict[str, Any]] = []

        async def nxt(ctx: Any) -> None:
            seen.append(dict(ctx.arguments))

        ctx = MafContext(MafFunction("read"), {"p": "danger"})
        run(ContractFunctionMiddleware(bridge([modify({"p": "safe"}), allow()])).process(ctx, nxt))
        assert seen[0] == {"p": "safe"}

    def test_denied_result_is_withheld(self) -> None:
        async def nxt(ctx: Any) -> None:
            ctx.result = "sk-secret"

        ctx = MafContext(MafFunction("read"))
        run(ContractFunctionMiddleware(bridge([allow(), deny(reason="leak")])).process(ctx, nxt))
        assert "sk-secret" not in ctx.result
        assert "withheld" in ctx.result

    def test_callable_form_is_the_same_pipeline(self) -> None:
        async def nxt(ctx: Any) -> None:
            ctx.result = "R"

        ctx = MafContext(MafFunction("read"))
        run(ContractFunctionMiddleware(bridge([allow(), allow()]))(ctx, nxt))
        assert ctx.result == "R"

    def test_function_name_falls_back_to_a_flat_attribute(self) -> None:
        @dataclass
        class FlatCtx:
            function_name: str
            arguments: dict[str, Any] = field(default_factory=dict)
            result: Any = None

        async def nxt(ctx: Any) -> None:
            ctx.result = "R"

        enforcer = StubEnforcer([allow(), allow()])
        run(ContractFunctionMiddleware(EnforcementBridge(enforcer)).process(FlatCtx("read"), nxt))
        assert enforcer.events[0].tool == "read"


# ---------------------------------------------------------------------------
# AgentScope — pre-hooks raise to abort the act
# ---------------------------------------------------------------------------


class FakeAgentScopeAgent:
    def __init__(self) -> None:
        self.hooks: dict[str, Any] = {}

    def register_instance_hook(self, event: str, name: str, fn: Any) -> None:
        self.hooks[event] = (name, fn)


class TestAgentScope:
    def test_allowed_act_returns_none_to_leave_kwargs_alone(self) -> None:
        hook = agentscope_pre_acting_hook(bridge([allow()]))
        assert hook(None, {"tool_call": {"name": "read", "input": {"p": 1}}}) is None

    def test_denied_act_raises_contract_breach(self) -> None:
        # AgentScope pre-hooks have no False convention; raising is how the act
        # is aborted, and it carries the reason.
        hook = agentscope_pre_acting_hook(bridge([deny(reason="destructive")]))
        with pytest.raises(ContractBreachError) as exc:
            hook(None, {"tool_call": {"name": "rm", "input": {}}})
        assert "destructive" in str(exc.value)

    def test_modify_returns_rewritten_kwargs(self) -> None:
        hook = agentscope_pre_acting_hook(bridge([modify({"p": "safe"})]))
        out = hook(None, {"tool_call": {"name": "read", "input": {"p": "danger"}}})
        assert out["tool_call"]["input"] == {"p": "safe"}

    def test_flat_kwargs_form_is_understood(self) -> None:
        enforcer = StubEnforcer([allow()])
        agentscope_pre_acting_hook(EnforcementBridge(enforcer))(
            None, {"name": "read", "input": {"p": 1}}
        )
        assert enforcer.events[0].tool == "read"
        assert enforcer.events[0].args == {"p": 1}

    def test_block_object_form_is_understood(self) -> None:
        @dataclass
        class ToolUseBlock:
            name: str
            input: dict[str, Any]

        enforcer = StubEnforcer([allow()])
        agentscope_pre_acting_hook(EnforcementBridge(enforcer))(
            None, {"block": ToolUseBlock("read", {"p": 1})}
        )
        assert enforcer.events[0].tool == "read"

    def test_kwargs_without_a_tool_call_are_skipped(self) -> None:
        enforcer = StubEnforcer()
        assert agentscope_pre_acting_hook(EnforcementBridge(enforcer))(None, {}) is None
        assert enforcer.events == []

    def test_post_hook_raises_on_a_denied_result(self) -> None:
        hook = agentscope_post_acting_hook(bridge([deny(reason="leaked")]))
        with pytest.raises(ContractBreachError):
            hook(None, {"name": "read", "input": {}}, "sk-secret")

    def test_post_hook_returns_the_output_when_clean(self) -> None:
        hook = agentscope_post_acting_hook(bridge([allow()]))
        assert hook(None, {"name": "read", "input": {}}, "fine") == "fine"

    def test_register_attaches_both_acting_hooks(self) -> None:
        agent = FakeAgentScopeAgent()
        register_agentscope_hooks(agent, bridge())
        assert set(agent.hooks) == {"pre__acting", "post__acting"}
        assert agent.hooks["pre__acting"][0] == "agentassert_contract"

    def test_register_rejects_an_object_without_the_hook_api(self) -> None:
        with pytest.raises(TypeError, match="register_instance_hook"):
            register_agentscope_hooks(object(), bridge())

    def test_modify_on_a_block_object_rewrites_its_input(self) -> None:
        @dataclass
        class ToolUseBlock:
            name: str
            input: dict[str, Any]

        block = ToolUseBlock("read", {"p": "danger"})
        out = agentscope_pre_acting_hook(bridge([modify({"p": "safe"})]))(None, {"block": block})
        assert out["block"].input == {"p": "safe"}

    def test_modify_on_flat_kwargs_sets_input(self) -> None:
        out = agentscope_pre_acting_hook(bridge([modify({"p": "safe"})]))(
            None, {"name": "read", "input": {"p": "danger"}}
        )
        assert out["input"] == {"p": "safe"}

    def test_modify_on_an_immutable_block_degrades_to_the_original(self) -> None:
        @dataclass(frozen=True)
        class FrozenBlock:
            name: str
            input: dict[str, Any]

        kwargs = {"block": FrozenBlock("read", {"p": "danger"})}
        out = agentscope_pre_acting_hook(bridge([modify({"p": "safe"})]))(None, kwargs)
        assert out is kwargs

    def test_post_hook_redacts_the_output(self, monkeypatch) -> None:
        import agentassert_abc.enforce.bridge as mod

        monkeypatch.setattr(mod, "apply_pii_redaction", lambda _t, _p: "[MASKED]")
        from ..test_mcp.conftest import redact

        hook = agentscope_post_acting_hook(bridge([redact()]))
        assert hook(None, {"name": "read", "input": {}}, "secret") == "[MASKED]"


class TestDegradationPaths:
    """A framework renaming a field must not crash an agent run."""

    def test_langchain_uses_the_frameworks_replace_helper_when_present(self) -> None:
        # Preferred over mutation so the rest of the request survives intact.
        seen: list[Any] = []

        class Replaceable:
            def __init__(self) -> None:
                self.tool_call = {"name": "read", "args": {"p": "danger"}, "id": "c"}

            def replace(self, **kw: Any) -> Any:
                other = Replaceable()
                other.tool_call = kw["tool_call"]
                return other

        wrap = langchain_tool_middleware(
            bridge([modify({"p": "safe"}), allow()]), message_factory=fake_message
        )
        wrap(Replaceable(), lambda r: seen.append(r) or "R")
        assert seen[0].tool_call["args"] == {"p": "safe"}

    def test_langchain_falls_back_to_mutation_when_replace_rejects_the_kwarg(self) -> None:
        seen: list[Any] = []

        class PickyReplace:
            def __init__(self) -> None:
                self.tool_call = {"name": "read", "args": {"p": "danger"}, "id": "c"}

            def replace(self, **kw: Any) -> Any:
                raise TypeError("unexpected keyword")

        wrap = langchain_tool_middleware(
            bridge([modify({"p": "safe"}), allow()]), message_factory=fake_message
        )
        wrap(PickyReplace(), lambda r: seen.append(r) or "R")
        assert seen[0].tool_call["args"] == {"p": "safe"}

    def test_langchain_modify_on_a_flat_request(self) -> None:
        @dataclass
        class Flat:
            name: str
            args: dict[str, Any]
            id: str

        seen: list[Any] = []
        wrap = langchain_tool_middleware(
            bridge([modify({"p": "safe"}), allow()]), message_factory=fake_message
        )
        wrap(Flat("read", {"p": "danger"}, "c"), lambda r: seen.append(r) or "R")
        assert seen[0].args == {"p": "safe"}

    def test_langchain_redacts_the_result(self, monkeypatch) -> None:
        import agentassert_abc.enforce.bridge as mod

        monkeypatch.setattr(mod, "apply_pii_redaction", lambda _t, _p: "[MASKED]")
        from ..test_mcp.conftest import redact

        wrap = langchain_tool_middleware(bridge([allow(), redact()]), message_factory=fake_message)
        out = wrap(LcRequest({"name": "read", "args": {}, "id": "c"}), lambda _r: "secret")
        assert out["content"] == "[MASKED]"

    def test_crewai_read_only_context_does_not_raise(self) -> None:
        class ReadOnly:
            tool_name = "read"

            @property
            def tool_input(self) -> dict[str, Any]:
                return {"p": 1}

            @property
            def tool_result(self) -> str:
                return "r"

        ctx = ReadOnly()
        assert crewai_before_tool_hook(bridge([modify({"p": 2})]))(ctx) is True
        assert crewai_after_tool_hook(bridge([deny(reason="x")]))(ctx) is False

    def test_crewai_context_without_any_reason_field_still_blocks(self) -> None:
        @dataclass
        class Minimal:
            tool_name: str
            tool_input: dict[str, Any] = field(default_factory=dict)

        assert crewai_before_tool_hook(bridge([deny()]))(Minimal("rm")) is False

    def test_crewai_redacts_the_result(self, monkeypatch) -> None:
        import agentassert_abc.enforce.bridge as mod

        monkeypatch.setattr(mod, "apply_pii_redaction", lambda _t, _p: "[MASKED]")
        from ..test_mcp.conftest import redact

        ctx = CrewContext("read", {}, tool_result="secret")
        crewai_after_tool_hook(bridge([redact()]))(ctx)
        assert ctx.tool_result == "[MASKED]"

    def test_maf_context_without_terminate_still_blocks(self) -> None:
        @dataclass
        class NoTerminate:
            function: MafFunction
            arguments: dict[str, Any] = field(default_factory=dict)
            result: Any = None

        called: list[Any] = []

        async def nxt(ctx: Any) -> None:
            called.append(ctx)

        ctx = NoTerminate(MafFunction("rm"))
        run(ContractFunctionMiddleware(bridge([deny(reason="x")])).process(ctx, nxt))
        assert called == []
        assert "blocked" in ctx.result

    def test_maf_read_only_context_does_not_raise(self) -> None:
        class ReadOnly:
            function = MafFunction("read")

            @property
            def arguments(self) -> dict[str, Any]:
                return {"p": 1}

            @property
            def result(self) -> str:
                return "r"

        async def nxt(_ctx: Any) -> None:
            return

        run(
            ContractFunctionMiddleware(bridge([modify({"p": 2}), allow()])).process(
                ReadOnly(), nxt
            )
        )

    def test_maf_unevaluated_call_skips_result_scoring(self) -> None:
        # Same contract as the LangChain shim: a fail-open pre-hook must not be
        # followed by scoring, which would blame the agent for our failure.
        async def nxt(ctx: Any) -> None:
            ctx.result = "R"

        ctx = MafContext(MafFunction("read"))
        run(
            ContractFunctionMiddleware(
                EnforcementBridge(StubEnforcer(raises=RuntimeError("boom")))
            ).process(ctx, nxt)
        )
        assert ctx.result == "R"

    def test_maf_redacts_the_result(self, monkeypatch) -> None:
        import agentassert_abc.enforce.bridge as mod

        monkeypatch.setattr(mod, "apply_pii_redaction", lambda _t, _p: "[MASKED]")
        from ..test_mcp.conftest import redact

        async def nxt(ctx: Any) -> None:
            ctx.result = "secret"

        ctx = MafContext(MafFunction("read"))
        run(ContractFunctionMiddleware(bridge([allow(), redact()])).process(ctx, nxt))
        assert ctx.result == "[MASKED]"
