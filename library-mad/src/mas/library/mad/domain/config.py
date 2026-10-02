"""Configuration models for one MAD runtime.

``MadConfig`` is the public entry point.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from mas.library.mad.domain.attacks import Attack, AttackPlacement
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Attack",
    "AttackPlacement",
    "DebateDataset",
    "DebateDecisions",
    "DebateFacts",
    "LLMSettings",
    "DebatePanel",
    "DebateProtocol",
    "MadConfig",
    "RunSettings",
]


# Supporting configuration types


class DebateFacts(StrEnum):
    """How the task's content is distributed across the panel."""

    SHARED = "shared"
    PARTITIONED = "partitioned"


class StructuredOutput(StrEnum):
    """How strictly each turn must arrive as a JSON envelope."""

    OFF = "off"
    BEST_EFFORT = "best_effort"
    REQUIRED = "required"

    @property
    def enabled(self) -> bool:
        return self is not StructuredOutput.OFF


class DebateDecisions(StrEnum):
    """Where an agent's commitment to an answer comes from."""

    PER_ROUND = "per_round"
    ENDPOINTS = "endpoints"


class DebatePanel(BaseModel):
    """Panel size, models, and role assignment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int | None = None
    models: tuple[str, ...]
    roles: str = "none"
    shuffle_roles: bool = False

    @model_validator(mode="after")
    def _check(self) -> "DebatePanel":
        if self.n is not None and self.n < 2:
            raise ValueError("panel.n must be at least 2")
        from mas.library.mad.domain.setup import is_valid_profile, profile_names

        if not is_valid_profile(self.roles):
            raise ValueError(
                f"panel.roles={self.roles!r} is not a known profile; expected one of "
                f"{profile_names()}"
            )
        if not self.models:
            raise ValueError("panel.models cannot be empty")
        if any(not model.strip() for model in self.models):
            raise ValueError("panel.models cannot contain blank model ids")
        return self


class DebateDataset(BaseModel):
    """Input dataset and optional sample limit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: Path
    n_samples: int | None = None

    @model_validator(mode="after")
    def _check(self) -> "DebateDataset":
        if self.n_samples is not None and self.n_samples < 1:
            raise ValueError("dataset.n_samples must be positive or null")
        return self


class DebateProtocol(BaseModel):
    """Fact sharing, decision collection, topology, and round scheduling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    facts: DebateFacts
    decisions: DebateDecisions
    rounds: int
    topology: str = "complete"
    random_sparsity: float = 1.0
    seed: int = 42
    discussion_max_workers: int = 1

    @model_validator(mode="after")
    def _check(self) -> "DebateProtocol":
        if self.rounds < 1:
            raise ValueError("protocol.rounds must be at least 1")
        if self.topology != "complete":
            # ponytail: the mas-lab port only implements complete-graph
            # visibility (see plugins/mad_debate.py); restore domain/topology.py
            # from mad_attack_env if another topology is ever needed.
            raise ValueError(
                f"protocol.topology={self.topology!r} is not supported by the mas-lab "
                "port; only 'complete' is implemented"
            )
        if not 0.0 <= self.random_sparsity <= 1.0:
            raise ValueError("protocol.random_sparsity must be in [0, 1]")
        if self.discussion_max_workers < 1:
            raise ValueError("protocol.discussion_max_workers must be at least 1")
        return self


class RunSettings(BaseModel):
    """Output location, parallelism, and failure handling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_dir: Path = Path("output")
    max_workers: int = 4
    name: str | None = None
    skip_failed_tasks: bool = False

    @model_validator(mode="after")
    def _check(self) -> "RunSettings":
        if self.max_workers < 1:
            raise ValueError("run.max_workers must be at least 1")
        return self


class LLMSettings(BaseModel):
    """Bounded, explicit gateway request and outer-retry policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_response_tokens: int | None = 4000
    """Cap on tokens a single debate turn may generate.

    Bounds runaway generation at the source rather than detecting it after the
    fact: a constrained-decoding failure was observed emitting ~63,000 spaces
    after a complete 2,107-char message. Typical turns run well under 1,000
    tokens, so the default leaves generous headroom. ``None`` removes the cap.
    """
    structured_output: StructuredOutput = StructuredOutput.OFF
    """How strictly each turn must come back as a JSON envelope.

    ``off`` (default) keeps the tagged free-text contract, so existing runs
    stay byte-comparable. ``best_effort`` sends a per-turn ``json_schema`` and
    splits the reply by field, falling back to the text path for any provider
    that ignores ``response_format`` -- every fallback is counted and reported
    so a half-constrained run is visible rather than silent. ``required``
    fails the task instead of falling back, for cohorts that must be uniformly
    schema-constrained.
    """
    request_timeout_seconds: float = 30.0
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 1.0

    @model_validator(mode="after")
    def _check(self) -> "LLMSettings":
        if self.max_response_tokens is not None and self.max_response_tokens < 1:
            raise ValueError("llm.max_response_tokens must be at least 1 when set")
        if self.request_timeout_seconds <= 0:
            raise ValueError("llm.request_timeout_seconds must be positive")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("llm.max_attempts must be between 1 and 3")
        if self.initial_backoff_seconds < 0:
            raise ValueError("llm.initial_backoff_seconds must be non-negative")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "llm.max_backoff_seconds must be at least initial_backoff_seconds"
            )
        return self


# Main configuration


class MadConfig(BaseModel):
    """One multi-agent debate run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: DebateDataset
    protocol: DebateProtocol
    panel: DebatePanel
    attack: Attack = Field(default_factory=Attack)
    run: RunSettings = Field(default_factory=RunSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @model_validator(mode="after")
    def _check(self) -> "MadConfig":
        attack_placement = (
            AttackPlacement.ADDITIVE
            if self.protocol.facts is DebateFacts.PARTITIONED
            else AttackPlacement.REPLACE
        )
        if self.attack.shuffle_seats and attack_placement is not AttackPlacement.ADDITIVE:
            raise ValueError(
                "attack.shuffle_seats=true requires an additive partitioned-facts attack"
            )
        if self.protocol.facts is DebateFacts.SHARED:
            if self.panel.n is None:
                raise ValueError("panel.n is required when protocol.facts is 'shared'")
        elif self.panel.n is not None:
            raise ValueError(
                "panel.n must be omitted when protocol.facts is 'partitioned' -- the "
                "panel size is derived from the task's hidden-fact count"
            )

        if self.panel.n is not None:
            total_seats = self.panel.n + (
                len(self.attack.seats)
                if self.attack.active and attack_placement is AttackPlacement.ADDITIVE
                else 0
            )
            if len(self.panel.models) not in (1, total_seats):
                raise ValueError(
                    f"panel.models has {len(self.panel.models)} entries but this configuration "
                    f"seats {total_seats} agents; provide one model or one per seat"
                )
            if self.attack.active:
                if (
                    self.attack.placement is not None
                    and self.attack.placement is not attack_placement
                ):
                    raise ValueError(
                        f"{self.protocol.facts} facts require attack.placement="
                        f"{attack_placement.value!r}"
                    )
                if attack_placement is AttackPlacement.ADDITIVE:
                    expected = tuple(range(self.panel.n, total_seats))
                    if self.attack.seats != expected:
                        raise ValueError(
                            "additive attack.seats must append after the cooperative panel: "
                            f"expected {list(expected)}"
                        )
                elif any(seat >= self.panel.n for seat in self.attack.seats):
                    raise ValueError(f"attack.seats must all be below panel.n ({self.panel.n})")
        elif (
            self.attack.active
            and self.attack.placement is not None
            and self.attack.placement is not attack_placement
        ):
            raise ValueError(
                f"{self.protocol.facts} facts require attack.placement={attack_placement.value!r}"
            )
        return self
