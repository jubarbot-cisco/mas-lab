#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Compile one MAD problem into a runnable mas-lab application.

MAD resolves a problem into seats, private facts, models and an attack.
That resolution is the experiment, so it is reused verbatim; this module
only decides where each resolved value belongs in a manifest.

Topology is not part of this: the port only implements MAD's complete graph
(every seat hears every other seat), enforced by ``MadConfig`` and by
``mad_debate`` directly. No adjacency is compiled or passed around.

Where things go, and why:

* seat system prompt -> that seat's ``spec.context``. A private fact reaches
  exactly one agent file, never the MAS manifest or the moderator.
* protocol and reminders -> moderator ``design_pattern.params``. The
  moderator is deterministic and never prompts a model with them.
* gold answer, attack metadata, fact assignments -> ``labels.json``, outside
  the manifests entirely. Nothing the runtime loads can read them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from mas.library.mad.domain.artifacts import seat_id
from mas.library.mad.domain.config import MadConfig
from mas.library.mad.domain.problems import DebateProblem
from mas.library.mad.domain.setup import build_setup

API_VERSION = "mas/v1"
MODERATOR_ID = "moderator"
LABELS_FILENAME = "labels.json"
MAS_FILENAME = "mas.yaml"

__all__ = ["CompiledApp", "compile_problem", "MODERATOR_ID", "LABELS_FILENAME", "MAS_FILENAME"]


@dataclass(frozen=True)
class CompiledApp:
    """Where a compiled problem landed on disk."""

    root: Path
    mas_manifest: Path
    labels: Path
    seats: tuple[str, ...]


def compile_problem(
    config: MadConfig,
    problem: DebateProblem,
    *,
    out_dir: Path,
    moderator_model: str | None = None,
) -> CompiledApp:
    """Write a self-contained MAS application for one debate."""
    setup = build_setup(config, problem)
    seats = tuple(seat_id(agent.id) for agent in setup.agents)
    names = [agent.name for agent in setup.agents]
    attackers = (
        tuple(seat_id(index) for index in sorted(setup.attack.seats)) if setup.attack else ()
    )

    root = Path(out_dir)
    (root / "agents").mkdir(parents=True, exist_ok=True)

    for agent in setup.agents:
        _write_yaml(
            root / "agents" / f"{seat_id(agent.id)}.yaml",
            _seat_manifest(
                seat=seat_id(agent.id),
                model=agent.model,
                system_prompt=agent.history[0]["content"],
                attacker=seat_id(agent.id) in attackers,
                max_tokens=config.llm.max_response_tokens,
            ),
        )

    moderator_params = {
        "task_id": problem.task_id,
        "question": problem.question,
        "options": list(problem.options),
        "seats": list(seats),
        "names": names,
        "rounds": config.protocol.rounds,
        "decisions": config.protocol.decisions.value,
        "attacker_seats": list(attackers),
        "reminders": (
            {seat_id(index): text for index, text in setup.attack.reminders.items()}
            if setup.attack
            else {}
        ),
        "warmup_rounds": setup.attack.warmup_rounds if setup.attack else 0,
        "structured": config.llm.structured_output.enabled,
    }
    _write_yaml(
        root / "agents" / f"{MODERATOR_ID}.yaml",
        _moderator_manifest(moderator_params, model=moderator_model),
    )

    mas_manifest = root / MAS_FILENAME
    _write_yaml(mas_manifest, _mas_manifest(problem, seats))

    labels = root / LABELS_FILENAME
    labels.write_text(
        json.dumps(_labels(config, problem, setup, seats, attackers), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return CompiledApp(root=root, mas_manifest=mas_manifest, labels=labels, seats=seats)


# --- manifests -------------------------------------------------------------


def _mas_manifest(problem: DebateProblem, seats: tuple[str, ...]) -> dict[str, Any]:
    return {
        "apiVersion": API_VERSION,
        "kind": "MAS",
        "metadata": {"name": _app_name(problem)},
        "spec": {
            "agency": {
                "agents": [{"id": MODERATOR_ID, "ref": f"agents/{MODERATOR_ID}.yaml"}]
                + [{"id": seat, "ref": f"agents/{seat}.yaml"} for seat in seats]
            },
            "workflow": {
                "entry": MODERATOR_ID,
                "nodes": [{"id": MODERATOR_ID, "delegates_to": list(seats)}]
                + [{"id": seat} for seat in seats],
            },
        },
    }


def _seat_manifest(
    *,
    seat: str,
    model: str,
    system_prompt: str,
    attacker: bool,
    max_tokens: int | None,
) -> dict[str, Any]:
    model_entry: dict[str, Any] = {"model": model}
    if max_tokens is not None:
        model_entry["max_tokens"] = int(max_tokens)
    return {
        "apiVersion": API_VERSION,
        "kind": "Agent",
        "metadata": {"name": seat},
        "spec": {
            "description": f"MAD debate participant {seat}.",
            "models": [model_entry],
            "design_pattern": {
                "type": "mad_turn",
                "params": {"seat_id": seat, "attacker": attacker},
            },
            # Private planning must not be quoted back into this seat's own
            # later prompts; mad_redact strips it on the way in.
            "context_manager": {"type": "mad_redact"},
            "context": {"mad_brief": system_prompt},
            "working_memory": {"persistent": True},
        },
    }


def _moderator_manifest(params: dict[str, Any], *, model: str | None) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "description": "Deterministic MAD protocol coordinator.",
        "design_pattern": {"type": "mad_debate", "params": params},
    }
    if model:
        spec["models"] = [{"model": model}]
    return {
        "apiVersion": API_VERSION,
        "kind": "Agent",
        "metadata": {"name": MODERATOR_ID},
        "spec": spec,
    }


def _labels(
    config: MadConfig,
    problem: DebateProblem,
    setup: Any,
    seats: tuple[str, ...],
    attackers: tuple[str, ...],
) -> dict[str, Any]:
    """Restricted sidecar: everything an evaluator needs and an agent must not see."""
    attack = setup.attack
    return {
        "schema_version": "mad-maslab-labels/v1",
        "benchmark_task_id": problem.task_id,
        "correct_option_id": list(problem.options).index(problem.correct),
        "model_by_seat_id": {seat_id(agent.id): agent.model for agent in setup.agents},
        "attack": {
            "present": attack is not None,
            "mode": attack.mode if attack else None,
            "placement": (attack.placement.value if attack else None),
            "adversary_seat_ids": list(attackers),
            "target_option_id": (
                list(problem.options).index(attack.target)
                if attack and attack.target in problem.options
                else None
            ),
            "warmup_rounds": attack.warmup_rounds if attack else 0,
        },
        "fact_assignments": [agent.view.to_dict() for agent in setup.agents],
        "seat_assignment": dict(setup.seat_assignment),
        "resolved_config": config.model_dump(mode="json"),
    }


def _app_name(problem: DebateProblem) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in problem.task_id)
    return f"mad-{safe}".strip("-").lower()


def _write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
