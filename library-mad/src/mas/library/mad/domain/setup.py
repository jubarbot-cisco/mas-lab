"""Prepare views, roles, prompts, and topology for one debate."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

import yaml
from mas.library.mad.domain import attacks
from mas.library.mad.domain.attacks import AttackPlacement, AttackSpec
from mas.library.mad.domain.config import DebateFacts, MadConfig
from mas.library.mad.domain.conversation import build_system_prompt
from mas.library.mad.domain.problems import DebateProblem
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "Agent",
    "AgentView",
    "DebateSetup",
    "build_views",
    "build_setup",
    "is_valid_profile",
    "panel_size",
    "profile_names",
    "role_for",
]

# Keep per-task random streams separated. This matches the former engine seed
# derivation so existing fact assignments and attack targets remain reproducible.
_TASK_STRIDE = 10_000
_AGENT_STREAM_STRIDE = 1_000_003
_ROLE_STREAM = 97
_NO_PERSONA = frozenset({"none", "diverse-models-only"})


@lru_cache(maxsize=1)
def _asymmetric_role_catalog() -> dict[str, list[dict[str, Any]]]:
    """Load the bundled asymmetric role profiles once per process."""
    resource = files("mas.library.mad.domain") / "content.yaml"
    with resource.open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    return catalog["asymmetric"]


_PROFILES: dict[str, tuple[tuple[str, str], ...]] = {
    name: tuple((role["name"], role["prompt"]) for role in profile)
    for name, profile in _asymmetric_role_catalog().items()
}


class AgentView(BaseModel):
    """Information available to one agent before deliberation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: int = Field(ge=0)
    name: str = Field(min_length=1, serialization_alias="agent")
    facts: tuple[str, ...] = Field(default=(), serialization_alias="visible_facts")
    private: tuple[str, ...] = Field(default=(), serialization_alias="assigned_hidden_information")

    @field_validator("name")
    @classmethod
    def _name_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value

    def to_dict(self) -> dict[str, object]:
        """Return the stable trace representation."""
        return self.model_dump(mode="json", by_alias=True)


class Agent:
    """One participant and its live conversation."""

    __slots__ = ("history", "id", "model", "utterances", "view")

    def __init__(self, agent_id: int, model: str, view: AgentView, system_prompt: str) -> None:
        self.id = agent_id
        self.model = model
        self.view = view
        self.history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self.utterances: list[str] = []

    @property
    def name(self) -> str:
        """Return the prompt-facing agent name."""
        return self.view.name


@dataclass
class DebateSetup:
    """Live state and immutable setup results for one debate."""

    agents: list[Agent]
    attack: AttackSpec | None
    seat_assignment: dict[str, object]


def profile_names() -> tuple[str, ...]:
    """Return valid persona profile names."""
    return (*sorted(_NO_PERSONA), *sorted(_PROFILES))


def is_valid_profile(profile: str) -> bool:
    """Return whether a persona profile is recognized."""
    return profile in _NO_PERSONA or profile in _PROFILES


def role_for(profile: str, n: int, *, seed: int, shuffle: bool = False) -> list[str | None]:
    """Return one cooperative role prompt per seat."""
    if profile in _NO_PERSONA:
        return [None] * n
    available = _PROFILES.get(profile)
    if available is None:
        raise ValueError(f"unknown role profile {profile!r}; expected one of {profile_names()}")
    if n > len(available):
        raise ValueError(
            f"role profile {profile!r} defines {len(available)} roles but the panel seats {n}"
        )
    selected = list(available[:n])
    if shuffle:
        random.Random(seed + _ROLE_STREAM).shuffle(selected)
    return [f"Your role is {name}. {prompt}" for name, prompt in selected]


def build_setup(config: MadConfig, problem: DebateProblem) -> DebateSetup:
    """Build the deterministic panel state for one problem."""
    seed = _task_seed(config.protocol.seed, problem.index)
    n = panel_size(config, problem)
    holders = _cooperative_panel_size(config, problem)
    identity_attackers = frozenset(config.attack.seats) if config.attack.active else frozenset()
    identity_views = build_views(
        config,
        problem,
        seed=seed,
        widened=(
            identity_attackers
            if config.attack.active and attacks.widens_view(config.attack.mode)
            else frozenset()
        ),
    )
    identity_roles = [
        *role_for(config.panel.roles, holders, seed=seed, shuffle=config.panel.shuffle_roles),
        *([None] * (n - holders)),
    ]
    identity_models = _models(config, n)
    identity_to_seat = _seat_permutation(config, problem, n)
    if config.attack.active and _attack_placement(config) is AttackPlacement.ADDITIVE:
        expected_additive_seats = tuple(range(holders, n))
        if config.attack.seats != expected_additive_seats:
            raise ValueError(
                f"problem {problem.task_id!r}: additive attackers must occupy appended seats "
                f"{list(expected_additive_seats)}, not {list(config.attack.seats)}"
            )
    actual_attackers = tuple(identity_to_seat[index] for index in config.attack.seats)
    attack = _resolve_attack(config, problem, n, seed, actual_seats=actual_attackers)

    views_by_seat = {
        identity_to_seat[identity]: view.model_copy(
            update={
                "agent_id": identity_to_seat[identity],
                "name": _agent_name(identity_to_seat[identity]),
            }
        )
        for identity, view in enumerate(identity_views)
    }
    models_by_seat = {
        identity_to_seat[identity]: model for identity, model in enumerate(identity_models)
    }
    roles_by_seat = {
        identity_to_seat[identity]: role for identity, role in enumerate(identity_roles)
    }
    agents = [
        Agent(
            agent_id=seat,
            model=models_by_seat[seat],
            view=views_by_seat[seat],
            system_prompt=build_system_prompt(
                question=problem.question,
                view=views_by_seat[seat],
                facts=config.protocol.facts,
                role=None if (attack and seat in attack.instructions) else roles_by_seat[seat],
                attack=attack.instructions.get(seat) if attack else None,
            ),
        )
        for seat in range(n)
    ]
    return DebateSetup(
        agents=agents,
        attack=attack,
        seat_assignment={
            "algorithm": "stable-problem-permutation/v1",
            "shuffled": config.attack.shuffle_seats,
            "problem_index": problem.index,
            "problem_identity": problem.task_id,
            "participant_identity_to_seat": list(identity_to_seat),
            "configured_adversary_identity_indices": list(config.attack.seats),
        },
    )


def build_views(
    config: MadConfig,
    problem: DebateProblem,
    *,
    seed: int,
    widened: frozenset[int] = frozenset(),
) -> list[AgentView]:
    """Allocate the configured facts to deterministic agent views."""
    n = panel_size(config, problem)
    if config.protocol.facts is DebateFacts.SHARED:
        return [AgentView(agent_id=i, name=_agent_name(i)) for i in range(n)]
    private = list(problem.hidden_facts)
    random.Random(seed).shuffle(private)
    holders = _cooperative_panel_size(config, problem)
    views: list[AgentView] = []
    for agent_id in range(n):
        own = (private[agent_id],) if agent_id < holders else ()
        visible_private = tuple(private) if agent_id in widened else own
        facts = [*problem.shared_facts, *visible_private]
        random.Random(seed + (agent_id + 1) * _AGENT_STREAM_STRIDE).shuffle(facts)
        views.append(
            AgentView(
                agent_id=agent_id,
                name=_agent_name(agent_id),
                facts=tuple(facts),
                private=own,
            )
        )
    return views


def panel_size(config: MadConfig, problem: DebateProblem) -> int:
    """Return the total panel size after benchmark-specific attack placement."""
    holders = _cooperative_panel_size(config, problem)
    if config.attack.active and _attack_placement(config) is AttackPlacement.ADDITIVE:
        return holders + len(config.attack.seats)
    return holders


def _cooperative_panel_size(config: MadConfig, problem: DebateProblem) -> int:
    """Return the benchmark-defined cooperative panel before attackers are placed."""
    if config.protocol.facts is DebateFacts.SHARED:
        assert config.panel.n is not None
        return config.panel.n
    if len(problem.hidden_facts) < 2:
        raise ValueError(
            f"problem {problem.task_id!r}: information='partitioned' needs at least 2 "
            f"hidden facts to split across a panel, found {len(problem.hidden_facts)}"
        )
    return len(problem.hidden_facts)


def _attack_placement(config: MadConfig) -> AttackPlacement:
    """Derive attack placement from the information benchmark, never a convention."""
    return (
        AttackPlacement.ADDITIVE
        if config.protocol.facts is DebateFacts.PARTITIONED
        else AttackPlacement.REPLACE
    )


def _agent_name(agent_id: int) -> str:
    """Return the one-based prompt-facing agent name."""
    return f"Agent {agent_id + 1}"


def _task_seed(base_seed: int, task_index: int) -> int:
    """Return the seed for all deterministic choices for one task."""
    return base_seed + task_index * _TASK_STRIDE


def _seat_permutation(config: MadConfig, problem: DebateProblem, n: int) -> tuple[int, ...]:
    """Map stable participant identities to public seats for one problem."""
    identities = list(range(n))
    if not (config.attack.active and config.attack.shuffle_seats):
        return tuple(identities)
    material = (
        f"stable-problem-permutation/v1\0{config.protocol.seed}\0{problem.index}\0{problem.task_id}"
    ).encode("utf-8")
    identities.sort(
        key=lambda identity: (
            hashlib.sha256(material + b"\0" + str(identity).encode("ascii")).digest(),
            identity,
        )
    )
    identity_to_seat = [0] * n
    for seat, identity in enumerate(identities):
        identity_to_seat[identity] = seat
    return tuple(identity_to_seat)


def _resolve_attack(
    config: MadConfig,
    problem: DebateProblem,
    n: int,
    seed: int,
    *,
    actual_seats: tuple[int, ...],
) -> AttackSpec | None:
    """Resolve and validate the optional attack for a panel."""
    if not config.attack.active:
        return None
    placement = _attack_placement(config)
    if config.attack.placement is not None and config.attack.placement is not placement:
        raise ValueError(
            f"{config.protocol.facts} facts require attack.placement={placement.value!r}"
        )
    holders = _cooperative_panel_size(config, problem)
    expected_additive_seats = tuple(range(holders, n))
    if placement is AttackPlacement.ADDITIVE and config.attack.seats != expected_additive_seats:
        raise ValueError(
            f"problem {problem.task_id!r}: additive attackers must occupy appended seats "
            f"{list(expected_additive_seats)}, not {list(config.attack.seats)}"
        )
    if placement is AttackPlacement.REPLACE and any(seat >= n for seat in config.attack.seats):
        raise ValueError(
            f"problem {problem.task_id!r}: attack seats {sorted(config.attack.seats)} "
            f"exceed the panel size ({n})"
        )
    return attacks.resolve(
        mode=config.attack.mode,
        placement=placement,
        seats=actual_seats,
        target=config.attack.target,
        options=problem.options,
        correct=problem.correct,
        names=[f"Agent {i + 1}" for i in range(n)],
        seed=seed,
        warmup_rounds=config.attack.warmup_rounds,
    )


def _models(config: MadConfig, n: int) -> list[str]:
    """Resolve one model identifier for each panel seat."""
    models = config.panel.models
    if len(models) == 1:
        return [models[0]] * n
    if len(models) != n:
        raise ValueError(
            f"panel.models has {len(models)} entries but this problem seats {n} agents"
        )
    return list(models)
