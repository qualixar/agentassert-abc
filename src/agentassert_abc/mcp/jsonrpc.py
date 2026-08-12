# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com

"""JSON-RPC framing for MCP's stdio transport.

Deliberately does **not** depend on the ``mcp`` package. The guard recognises
exactly one method — ``tools/call`` — and relays every other message
byte-for-byte. Modelling the full protocol would couple the guard to a spec
revision it has no reason to track, and would break the guard every time the
protocol adds a capability that has nothing to do with tool execution.

MCP's stdio transport frames messages as newline-delimited JSON: one JSON-RPC
object per line, with no embedded newlines. That is the whole wire format this
module needs to know.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "TOOL_CALL_METHOD",
    "decode",
    "encode",
    "is_tool_call_request",
    "request_id",
    "result_text",
    "tool_call_arguments",
    "tool_call_name",
    "tool_error_result",
    "with_tool_arguments",
    "with_result_text",
]

TOOL_CALL_METHOD = "tools/call"


def decode(line: str) -> dict[str, Any] | None:
    """Parse one wire line into a JSON-RPC object.

    Returns:
        The decoded object, or ``None`` if the line is blank or not a JSON
        object. A ``None`` here means "relay this line untouched" — the guard
        never drops traffic it failed to understand.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        msg = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    return msg if isinstance(msg, dict) else None


def encode(message: dict[str, Any]) -> str:
    """Serialise a JSON-RPC object to one wire line.

    ``ensure_ascii`` keeps the payload newline-free for any input, which is what
    the stdio framing requires.
    """
    return json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"


def is_tool_call_request(message: dict[str, Any]) -> bool:
    """True for a ``tools/call`` request carrying an id.

    A ``tools/call`` without an id is a notification, which by JSON-RPC
    semantics expects no response. The guard cannot synthesise a DENY reply for
    one, so it is relayed rather than intercepted.
    """
    return message.get("method") == TOOL_CALL_METHOD and message.get("id") is not None


def request_id(message: dict[str, Any]) -> Any:
    """The JSON-RPC id, which may be an int or a str."""
    return message.get("id")


def tool_call_name(message: dict[str, Any]) -> str:
    params = message.get("params")
    if not isinstance(params, dict):
        return ""
    name = params.get("name", "")
    return name if isinstance(name, str) else ""


def tool_call_arguments(message: dict[str, Any]) -> dict[str, Any]:
    params = message.get("params")
    if not isinstance(params, dict):
        return {}
    args = params.get("arguments")
    return args if isinstance(args, dict) else {}


def with_tool_arguments(message: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``message`` with rewritten tool arguments (MODIFY).

    Builds new dicts rather than mutating: the caller still holds the original
    for the audit log, and an in-place rewrite would corrupt it.
    """
    params = message.get("params")
    new_params = dict(params) if isinstance(params, dict) else {}
    new_params["arguments"] = arguments
    return {**message, "params": new_params}


def tool_error_result(req_id: Any, text: str) -> dict[str, Any]:
    """A ``tools/call`` result marked as an error, for a DENY.

    Reported as an ``isError`` *result* rather than a JSON-RPC *error* on
    purpose. A protocol-level error reads to most clients as a transport fault
    and may be retried or surfaced as a crash; an ``isError`` result is handed
    to the model as tool output, so the agent reads why it was blocked and can
    choose a different action. Enforcement should redirect the agent, not break
    its session.
    """
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": True,
        },
    }


def result_text(message: dict[str, Any]) -> str:
    """Concatenated text blocks from a ``tools/call`` result."""
    result = message.get("result")
    if not isinstance(result, dict):
        return ""
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(part for part in parts if isinstance(part, str))


def with_result_text(message: dict[str, Any], text: str) -> dict[str, Any]:
    """Copy of ``message`` with every text block replaced by ``text`` (REDACT).

    The first text block carries the replacement and any further text blocks are
    emptied, so a secret split across blocks cannot survive redaction in a later
    one.
    """
    result = message.get("result")
    if not isinstance(result, dict):
        return message
    content = result.get("content")
    if not isinstance(content, list):
        return message

    new_content: list[Any] = []
    replaced = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            new_content.append({**block, "text": text if not replaced else ""})
            replaced = True
        else:
            new_content.append(block)
    if not replaced:
        new_content.append({"type": "text", "text": text})
    return {**message, "result": {**result, "content": new_content}}
