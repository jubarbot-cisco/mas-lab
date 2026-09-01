#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""HTTP surface over a live ``RunningMas`` — reach any agent, from any host.

Two adapters, one seam: plain REST for our own tooling, a hand-rolled A2A subset
(agent card + ``message/send``) so a foreign agent can interrogate ours with no
client code written here. Both call ``RunningMas.ask()``, so an agent answers
from its real controller, its real history, and the MAS-wide session — which is
the whole point when the caller is asking what it did earlier.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from mas.ctl.executor.running_mas import RunningMas
from mas.ctl.ui.turn_result import turn_to_agent_result

logger = logging.getLogger(__name__)

# JSON-RPC: -32700/-32600/-32602 are spec; -32001 is ours for "no such agent".
_RPC_PARSE = -32700
_RPC_INVALID_PARAMS = -32602
_RPC_INTERNAL = -32603
_RPC_UNKNOWN_AGENT = -32001


def _turn_body(agent_id: str, result) -> dict[str, Any]:
    """Turn outcome as a body. Always 200: a failing agent is evidence."""
    if result is None:
        return {
            "agent": agent_id,
            "status": "error",
            "text": "",
            "error": "turn raised",
        }
    if result.awaiting_hitl:
        return {
            "agent": agent_id,
            "status": "awaiting_hitl",
            "text": result.text,
            "error": "turn is waiting on human approval",
        }
    outcome = turn_to_agent_result(result)
    return {
        "agent": agent_id,
        "status": outcome.status,
        "text": outcome.response,
        "error": outcome.error_message or None,
    }


def build_app(
    mas: RunningMas,
    cards: dict[str, dict[str, Any]],
    *,
    token: str | None = None,
) -> FastAPI:
    """FastAPI app serving ``mas``. ``token`` None disables auth."""
    app = FastAPI(title="mas-ctl serve-mas", version="0.1.0")
    # SessionController is sync and not thread-safe. One lock per agent: turns
    # to the same agent serialize (one conversation advances once at a time),
    # turns to different agents run in parallel.
    locks: dict[str, asyncio.Lock] = {a: asyncio.Lock() for a in mas.agent_ids}

    def require_auth(request: Request) -> None:
        if token is None:
            return
        header = request.headers.get("authorization", "")
        scheme, _, value = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(value, token):
            raise HTTPException(status_code=401, detail="bearer token required")

    auth = [Depends(require_auth)]

    def known(agent_id: str) -> str:
        if agent_id not in mas.agent_ids:
            raise HTTPException(status_code=404, detail=f"unknown agent '{agent_id}'")
        return agent_id

    async def ask(agent_id: str, text: str) -> Any:
        """One turn. Returns TurnResult, or None if it raised."""
        async with locks[agent_id]:
            try:
                return await run_in_threadpool(mas.ask, text, agent=agent_id)
            except Exception:  # one poisoned turn must not end the server
                logger.exception("turn to agent '%s' raised", agent_id)
                return None

    def check_text(text: Any) -> str:
        text = str(text or "").strip()
        if not text:
            raise ValueError("empty text")
        # run_turn intercepts '/steer ', and OperatorConsole owns /reset /abort
        # etc. A remote client gets to ask, never to steer.
        if text.startswith("/"):
            raise ValueError("control commands are not available over HTTP")
        return text

    # ---- REST -------------------------------------------------------------

    @app.get("/agents", dependencies=auth)
    async def list_agents() -> list[dict[str, str]]:
        return [
            {
                "id": agent_id,
                "url": url,
                "card": f"{url}.well-known/agent-card.json",
            }
            for agent_id in sorted(mas.agent_ids)
            if (url := (cards.get(agent_id) or {}).get("url"))
        ]

    @app.post("/agents/{agent_id}/ask", dependencies=auth)
    async def ask_agent(agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        known(agent_id)
        try:
            text = check_text(body.get("text"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = await ask(agent_id, text)
        return _turn_body(agent_id, result)

    # ---- A2A --------------------------------------------------------------

    @app.get("/agents/{agent_id}/.well-known/agent-card.json", dependencies=auth)
    async def agent_card(agent_id: str) -> dict[str, Any]:
        return cards[known(agent_id)]

    @app.post("/agents/{agent_id}/", dependencies=auth)
    async def a2a_rpc(agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        rpc_id = body.get("id") if isinstance(body, dict) else None

        def error(code: int, message: str) -> dict[str, Any]:
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}

        if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
            return error(_RPC_PARSE, "expected a JSON-RPC 2.0 request")
        if agent_id not in mas.agent_ids:
            return error(_RPC_UNKNOWN_AGENT, f"unknown agent '{agent_id}'")
        if body.get("method") != "message/send":
            return error(_RPC_INVALID_PARAMS, "only 'message/send' is supported")

        message = (body.get("params") or {}).get("message") or {}
        parts = [
            str(p.get("text") or "")
            for p in message.get("parts") or []
            if isinstance(p, dict) and p.get("kind", "text") == "text"
        ]
        try:
            text = check_text("\n".join(parts))
        except ValueError as exc:
            return error(_RPC_INVALID_PARAMS, str(exc))

        result = await ask(agent_id, text)
        outcome = _turn_body(agent_id, result)
        failed = outcome["status"] != "ok"
        task_id = uuid.uuid4().hex
        task: dict[str, Any] = {
            "id": task_id,
            "contextId": mas.session_id,
            "kind": "task",
            "status": {"state": "failed" if failed else "completed"},
            "artifacts": [],
        }
        if failed:
            task["status"]["message"] = {
                "role": "agent",
                "parts": [{"kind": "text", "text": outcome["error"] or "turn failed"}],
                "kind": "message",
                "messageId": f"{task_id}-err",
            }
        else:
            task["artifacts"] = [
                {
                    "artifactId": f"{task_id}-answer",
                    "parts": [{"kind": "text", "text": outcome["text"]}],
                }
            ]
        return {"jsonrpc": "2.0", "id": rpc_id, "result": task}

    return app
