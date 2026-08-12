# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE

"""JSON-RPC framing for the MCP stdio transport."""

from __future__ import annotations

import json

import pytest

from agentassert_abc.mcp import jsonrpc

from .conftest import tool_call, tool_result


class TestDecode:
    @pytest.mark.parametrize("line", ["", "   ", "\n", "not json", "[1,2]", '"a string"', "42"])
    def test_unparseable_or_non_object_returns_none(self, line: str) -> None:
        # None means "relay untouched" — the guard never drops traffic it did
        # not understand.
        assert jsonrpc.decode(line) is None

    def test_decodes_object_ignoring_surrounding_whitespace(self) -> None:
        assert jsonrpc.decode('  {"a": 1}\n') == {"a": 1}


class TestEncode:
    def test_roundtrips(self) -> None:
        msg = tool_call(1, "read", {"path": "/tmp/x"})
        assert jsonrpc.decode(jsonrpc.encode(msg)) == msg

    def test_emits_exactly_one_line(self) -> None:
        # stdio framing is newline-delimited, so an embedded newline in the
        # payload would desynchronise the peer's parser for the rest of the run.
        encoded = jsonrpc.encode({"text": "a\nb\nc", "emoji": "→ 🙂"})
        assert encoded.endswith("\n")
        assert encoded.count("\n") == 1

    def test_preserves_non_ascii_through_a_roundtrip(self) -> None:
        assert jsonrpc.decode(jsonrpc.encode({"t": "→ 🙂"}))["t"] == "→ 🙂"


class TestIsToolCallRequest:
    def test_true_for_tool_call_with_id(self) -> None:
        assert jsonrpc.is_tool_call_request(tool_call(1, "read")) is True

    def test_id_zero_is_a_valid_request(self) -> None:
        # `if not message.get("id")` would wrongly treat id 0 as absent and let
        # a real call slip past unguarded.
        assert jsonrpc.is_tool_call_request(tool_call(0, "read")) is True

    def test_string_ids_are_valid(self) -> None:
        assert jsonrpc.is_tool_call_request(tool_call("abc", "read")) is True

    def test_false_without_id_because_a_notification_expects_no_reply(self) -> None:
        assert jsonrpc.is_tool_call_request({"method": "tools/call", "params": {}}) is False

    @pytest.mark.parametrize("method", ["tools/list", "initialize", "resources/read", ""])
    def test_false_for_other_methods(self, method: str) -> None:
        assert jsonrpc.is_tool_call_request({"method": method, "id": 1}) is False


class TestToolCallAccessors:
    def test_reads_name_and_arguments(self) -> None:
        msg = tool_call(1, "read_file", {"path": "/etc/passwd"})
        assert jsonrpc.tool_call_name(msg) == "read_file"
        assert jsonrpc.tool_call_arguments(msg) == {"path": "/etc/passwd"}

    @pytest.mark.parametrize(
        "params", [None, "string-params", 42, {}, {"name": 123}, {"arguments": "no"}]
    )
    def test_malformed_params_degrade_to_empty(self, params: object) -> None:
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params}
        assert jsonrpc.tool_call_name(msg) == ""
        assert jsonrpc.tool_call_arguments(msg) == {}


class TestWithToolArguments:
    def test_replaces_arguments_and_keeps_the_rest(self) -> None:
        msg = tool_call(7, "read_file", {"path": "/etc/passwd"})
        out = jsonrpc.with_tool_arguments(msg, {"path": "/tmp/safe"})
        assert out["params"]["arguments"] == {"path": "/tmp/safe"}
        assert out["params"]["name"] == "read_file"
        assert out["id"] == 7

    def test_does_not_mutate_the_original(self) -> None:
        # The caller still holds the original for the audit log; an in-place
        # rewrite would make the log report the sanitised args as the ones the
        # agent asked for.
        msg = tool_call(1, "read_file", {"path": "/etc/passwd"})
        jsonrpc.with_tool_arguments(msg, {"path": "/tmp/safe"})
        assert msg["params"]["arguments"] == {"path": "/etc/passwd"}

    def test_synthesises_params_when_absent(self) -> None:
        out = jsonrpc.with_tool_arguments({"id": 1, "method": "tools/call"}, {"a": 1})
        assert out["params"]["arguments"] == {"a": 1}


class TestToolErrorResult:
    def test_is_an_error_result_not_a_protocol_error(self) -> None:
        # Reported as `isError` so the model reads the block as tool output and
        # can adapt. A JSON-RPC `error` reads as a transport fault to most
        # clients and may be retried or surfaced as a crash.
        out = jsonrpc.tool_error_result(3, "blocked")
        assert "error" not in out
        assert out["id"] == 3
        assert out["result"]["isError"] is True
        assert out["result"]["content"][0]["text"] == "blocked"

    def test_survives_encoding(self) -> None:
        assert json.loads(jsonrpc.encode(jsonrpc.tool_error_result(1, "x")))["id"] == 1


class TestResultText:
    def test_concatenates_text_blocks(self) -> None:
        msg = {
            "id": 1,
            "result": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        }
        assert jsonrpc.result_text(msg) == "ab"

    def test_ignores_non_text_blocks(self) -> None:
        msg = {
            "id": 1,
            "result": {
                "content": [{"type": "image", "data": "..."}, {"type": "text", "text": "t"}]
            },
        }
        assert jsonrpc.result_text(msg) == "t"

    @pytest.mark.parametrize(
        "result", [None, "str", {"content": None}, {"content": "str"}, {}, {"content": []}]
    )
    def test_missing_or_malformed_content_is_empty(self, result: object) -> None:
        assert jsonrpc.result_text({"id": 1, "result": result}) == ""


class TestWithResultText:
    def test_first_text_block_carries_the_replacement(self) -> None:
        out = jsonrpc.with_result_text(tool_result(1, "secret"), "[REDACTED]")
        assert out["result"]["content"][0]["text"] == "[REDACTED]"

    def test_later_text_blocks_are_emptied(self) -> None:
        # A secret split across blocks must not survive in a later one.
        msg = {
            "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": "sec"},
                    {"type": "text", "text": "ret"},
                ]
            },
        }
        out = jsonrpc.with_result_text(msg, "[REDACTED]")
        assert [b["text"] for b in out["result"]["content"]] == ["[REDACTED]", ""]
        assert jsonrpc.result_text(out) == "[REDACTED]"

    def test_non_text_blocks_are_preserved(self) -> None:
        msg = {
            "id": 1,
            "result": {"content": [{"type": "image", "data": "d"}, {"type": "text", "text": "s"}]},
        }
        out = jsonrpc.with_result_text(msg, "[R]")
        assert out["result"]["content"][0] == {"type": "image", "data": "d"}

    def test_appends_a_block_when_the_result_had_no_text(self) -> None:
        msg = {"id": 1, "result": {"content": [{"type": "image", "data": "d"}]}}
        out = jsonrpc.with_result_text(msg, "[R]")
        assert jsonrpc.result_text(out) == "[R]"

    def test_does_not_mutate_the_original(self) -> None:
        msg = tool_result(1, "secret")
        jsonrpc.with_result_text(msg, "[R]")
        assert msg["result"]["content"][0]["text"] == "secret"

    @pytest.mark.parametrize("result", [None, "str", {"content": "notalist"}])
    def test_unshaped_result_is_returned_unchanged(self, result: object) -> None:
        msg = {"id": 1, "result": result}
        assert jsonrpc.with_result_text(msg, "[R]") == msg
