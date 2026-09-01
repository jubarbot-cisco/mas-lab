#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""MCP surface over a live ``RunningMas`` — invoke the agents' own tools.

An agent card states which tools an agent claims; this lets a client call one
and see what it actually returns. Same runtime, same instances, same state the
agents use — a value from a fresh instance would not be the value the agent saw.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_RPC_PARSE = -32700
_RPC_METHOD_NOT_FOUND = -32601
_RPC_INVALID_PARAMS = -32602


@dataclass
class ToolEntry:
    """One flat-set tool: what it claims, who declared it, how to run it.

    ``name`` is the tool's real name; the dict key it is filed under may differ
    (``name@agent``), so the key is for display and the name is what we call.
    """

    name: str
    description: str
    schema: dict[str, Any]
    declared_by: list[str]
    agent_id: str  # whose provider executes it
    provider: Any


def build_tools(mas: Any) -> dict[str, ToolEntry]:
    """Flat global tool set, from the live per-agent providers.

    Same name + same contract across N agents collapses to one entry (providers
    are per-agent but tools are stateless, so the instances are interchangeable);
    same name with a *different* contract is disambiguated as ``name@agent`` so
    that every tool of every agent stays reachable.
    """
    tools: dict[str, ToolEntry] = {}
    for agent_id in sorted(mas.agent_ids):
        provider = mas.tool_provider(agent_id)
        if provider is None:
            continue
        for spec in provider.list_tools():
            name = str(spec.get("name") or "")
            if not name:
                continue
            entry = ToolEntry(
                name,
                str(spec.get("description") or f"Invoke tool {name}."),
                spec.get("parameters") or {"type": "object", "properties": {}},
                [agent_id],
                agent_id,
                provider,
            )
            seen = tools.get(name)
            if seen is None:
                tools[name] = entry
            elif (seen.description, seen.schema) == (entry.description, entry.schema):
                seen.declared_by.append(agent_id)
            else:
                tools[f"{name}@{agent_id}"] = entry
    return tools


async def handle_rpc(body: Any, tools: dict[str, ToolEntry], call) -> dict[str, Any]:
    """One MCP JSON-RPC request. ``call(entry, arguments) -> str``.

    Stateless: ``initialize`` hands back no session id, and any sent is ignored —
    the state that matters lives in the MAS, not in the connection.
    """
    rpc_id = body.get("id") if isinstance(body, dict) else None

    def ok(result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rpc_id, "result": result}

    def fail(code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}

    def text(body_text: str, *, is_error: bool = False) -> dict[str, Any]:
        return ok({"content": [{"type": "text", "text": body_text}], "isError": is_error})

    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
        return fail(_RPC_PARSE, "expected a JSON-RPC 2.0 request")

    method = body.get("method")
    params = body.get("params") or {}

    match method:
        case "initialize":
            return ok(
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mas-ctl serve-mas", "version": "0.1.0"},
                }
            )
        case "tools/list":
            return ok(
                {
                    "tools": [
                        {
                            "name": key,
                            "description": e.description,
                            "inputSchema": e.schema,
                            "declared_by": e.declared_by,
                        }
                        for key, e in sorted(tools.items())
                    ]
                }
            )
        case "tools/call":
            pass
        case _:
            return fail(_RPC_METHOD_NOT_FOUND, f"unsupported method '{method}'")

    key = str(params.get("name") or "")
    entry = tools.get(key)
    if entry is None:
        return fail(_RPC_METHOD_NOT_FOUND, f"unknown tool '{key}'")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        return fail(_RPC_INVALID_PARAMS, "'arguments' must be an object")

    try:
        return text(await call(entry, arguments))
    except Exception as exc:  # a failing tool is a finding, not a transport fault
        logger.exception("tool '%s' raised", entry.name)
        return text(f"{type(exc).__name__}: {exc}", is_error=True)
