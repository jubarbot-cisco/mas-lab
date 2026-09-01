#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""A2A agent cards for the agents of a running MAS.

Built once at startup: ``RuntimeInstance`` does not keep its composed spec, so
each card is re-read from the agent's manifest on disk. The card describes what
an agent *declares* — role, tools, delegation peers — never what it did. The gap
between the claim and an actual reply is what an interrogating client compares.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mas.ctl.executor.mas_session import (
    MaterializedMas,
    agent_manifest_path,
    load_agent_manifest_from_bind,
)
from mas.ctl.manifest.mas_agent_merge import enrich_entry_agent_for_delegation
from mas.runtime.boundary.delegation.policy import delegation_targets

# Cards are metadata, not a payload: a whole system prompt as `description`
# helps nobody and some registries truncate it anyway.
_MAX_DESCRIPTION = 800


def _role_text(manifest: dict[str, Any]) -> str:
    role = ((manifest.get("spec") or {}).get("context") or {}).get("role")
    if isinstance(role, dict):  # {ref: path} — the file is the prompt, not the card
        role = ""
    text = " ".join(str(role or "").split())
    if not text:
        text = str((manifest.get("spec") or {}).get("description") or "")
    return text[:_MAX_DESCRIPTION]


def _tool_skills(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    skills = []
    for tool in (manifest.get("spec") or {}).get("tools") or []:
        if not isinstance(tool, dict):
            continue
        # ponytail: ref stem is the tool name in practice; resolving every
        # $ref just to title a skill is not worth the compose machinery.
        name = tool.get("name") or Path(str(tool.get("ref") or "")).name.split(".")[0]
        if not name:
            continue
        skills.append(
            {
                "id": f"tool:{name}",
                "name": str(name),
                "description": str(tool.get("description") or f"Declared tool {name}."),
                "tags": ["tool"],
            }
        )
    return skills


def _peer_skills(peers: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"delegate:{peer}",
            "name": f"delegate_to_{peer}",
            "description": f"Delegates to peer agent '{peer}'.",
            "tags": ["delegation"],
        }
        for peer in peers
    ]


def build_card(
    materialized: MaterializedMas,
    agent_id: str,
    *,
    base_url: str,
    secured: bool,
) -> dict[str, Any]:
    """One A2A agent card, read from the agent's manifest on disk."""
    compose = materialized.compose
    manifest = load_agent_manifest_from_bind(compose.bind, agent_id) or {}
    manifest_path = agent_manifest_path(compose.bind, agent_id)
    enriched = enrich_entry_agent_for_delegation(
        manifest,
        compose.mas_config,
        manifest_dir=manifest_path.parent if manifest_path else materialized.mas_base_dir,
        mas_base_dir=materialized.mas_base_dir,
    )
    peers = delegation_targets(enriched, agent_id=agent_id)

    card: dict[str, Any] = {
        "protocolVersion": "0.2.5",
        "name": agent_id,
        "description": _role_text(manifest) or f"Agent '{agent_id}' of a running MAS.",
        "url": f"{base_url}/agents/{agent_id}/",
        "version": str((manifest.get("metadata") or {}).get("version") or "0.1.0"),
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": _tool_skills(manifest) + _peer_skills(peers),
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
