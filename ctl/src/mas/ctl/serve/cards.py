#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""A2A agent cards for the agents of a running MAS.

Built once at startup: ``RuntimeInstance`` does not keep its composed spec, so
each card is re-read from the agent's manifest on disk. The card exposes only the
agent's declared ``spec.description``: never its role, context, prompt, tools or
delegation peers.
"""

from __future__ import annotations

from typing import Any

from mas.ctl.executor.mas_session import MaterializedMas, load_agent_manifest_from_bind

# Cards are metadata, not a payload; some registries truncate long descriptions.
_MAX_DESCRIPTION = 800


def _description(manifest: dict[str, Any]) -> str:
    text = " ".join(str((manifest.get("spec") or {}).get("description") or "").split())
    return text[:_MAX_DESCRIPTION]


def build_card(
    materialized: MaterializedMas,
    agent_id: str,
    *,
    base_url: str,
    secured: bool,
) -> dict[str, Any]:
    """One A2A agent card, read from the agent's manifest on disk."""
    manifest = load_agent_manifest_from_bind(materialized.compose.bind, agent_id) or {}

    card: dict[str, Any] = {
        "protocolVersion": "0.2.5",
        "name": agent_id,
        "description": _description(manifest) or f"Agent '{agent_id}' of a running MAS.",
        "url": f"{base_url}/agents/{agent_id}/",
        "version": str((manifest.get("metadata") or {}).get("version") or "0.1.0"),
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [],
    }
    if secured:
        card["securitySchemes"] = {"bearer": {"type": "http", "scheme": "bearer"}}
        card["security"] = [{"bearer": []}]
    return card


def build_cards(
    materialized: MaterializedMas,
    *,
    base_url: str,
    secured: bool,
) -> dict[str, dict[str, Any]]:
    """Card per materialized agent. Called once, at server startup."""
    return {
        agent_id: build_card(materialized, agent_id, base_url=base_url, secured=secured)
        for agent_id in materialized.materialized.instances
    }
