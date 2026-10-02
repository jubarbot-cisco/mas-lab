"""Versioned durable MAD output contracts and projection helpers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PUBLIC_TRACE_SCHEMA = "mad-public-trace/v1"
EVALUATOR_LABELS_SCHEMA = "mad-evaluator-labels/v1"
RUN_MANIFEST_SCHEMA = "mad-run-manifest/v3"
ATTACK_SCRATCHPAD_SCHEMA = "mad-attack-scratchpad/v1"


class _Artifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Option(_Artifact):
    id: int = Field(ge=0)
    text: str


class Decision(_Artifact):
    seat_id: str
    option_id: int | None = Field(default=None, ge=0)


class PeerTurnReference(_Artifact):
    seat_id: str
    round: int = Field(ge=1)


class PublicTurn(_Artifact):
    seat_id: str
    text: str
    decision_option_id: int | None = Field(default=None, ge=0)
    observed_peer_turns: tuple[PeerTurnReference, ...] = ()


class PublicRound(_Artifact):
    round: int = Field(ge=1)
    turns: tuple[PublicTurn, ...]


class DetectorTrace(_Artifact):
    schema_version: str = PUBLIC_TRACE_SCHEMA
    trace_id: str
    question: str
    options: tuple[Option, ...]
    participant_seat_ids: tuple[str, ...]
    initial_selected_option_ids: tuple[Decision, ...]
    final_selected_option_ids: tuple[Decision, ...]
    rounds: tuple[PublicRound, ...]


class AttackLabels(_Artifact):
    present: bool
    mode: str | None = None
    placement: str | None = None
    adversary_seat_ids: tuple[str, ...] = ()
    cooperative_fact_holder_seat_ids: tuple[str, ...] = ()
    target_option_id: int | None = Field(default=None, ge=0)
    warmup_rounds: int = Field(default=0, ge=0)
    """Rounds of trust-building actually applied, after any per-task range draw.

    Recorded so attack success can be correlated against how long the
    adversary waited before starting.
    """


class EvaluatorLabels(_Artifact):
    schema_version: str = EVALUATOR_LABELS_SCHEMA
    trace_id: str
    correct_option_id: int = Field(ge=0)
    attack: AttackLabels
    model_by_seat_id: dict[str, str]
    benchmark_task_id: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    fact_assignments: tuple[dict[str, Any], ...] = ()
    private_ballot_rationales: dict[str, dict[str, str]] = Field(default_factory=dict)


class ScratchpadEntry(_Artifact):
    """One attacker's private planning entry for one round, paired with what it said.

    Ground truth only: never derivable from ``DetectorTrace`` alone. ``parsed``
    is ``False`` when the model emitted a scratchpad block whose content was
    not valid JSON -- in that case only ``raw_text`` is reliable.
    ``discussion_text`` is the same public turn text this round's entry in
    ``traces/`` carries, included here too so a reader can see plan and
    execution together without joining two files.
    """

    seat_id: str
    round: int = Field(ge=1)
    goal: str | None = None
    reasoning_steps: tuple[str, ...] = ()
    planned_action: str | None = None
    raw_text: str
    parsed: bool
    discussion_text: str = ""


class AttackerScratchpadLog(_Artifact):
    """Restricted per-task log of every adversary seat's scratchpad entries.

    Written only for tasks with an active attack; a benign task has no
    scratchpad file at all. Linked to its trace and labels by ``trace_id``.
    """

    schema_version: str = ATTACK_SCRATCHPAD_SCHEMA
    trace_id: str
    benchmark_task_id: str
    entries: tuple[ScratchpadEntry, ...]


class FailedTask(_Artifact):
    """A task omitted from an incomplete, skip-failed batch."""

    task_index: int = Field(ge=0)
    error_type: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=200)


class CaseSeatAssignment(_Artifact):
    """Restricted per-case inventory of post-shuffle public seats."""

    task_index: int = Field(ge=0)
    trace_id: str
    benchmark_task_id: str
    adversary_seat_ids: tuple[str, ...] = ()
    cooperative_fact_holder_seat_ids: tuple[str, ...] = ()


class CallLatencyMetrics(_Artifact):
    """Aggregate wall-clock latency for one call category."""

    call_count: int = Field(default=0, ge=0)
    mean_wall_seconds: float = Field(default=0.0, ge=0.0)
    max_wall_seconds: float = Field(default=0.0, ge=0.0)


class ModelLatencyMetrics(_Artifact):
    """Aggregate-only latency attribution for one configured model."""

    total: CallLatencyMetrics = Field(default_factory=CallLatencyMetrics)
    normal: CallLatencyMetrics = Field(default_factory=CallLatencyMetrics)
    ballot_repair: CallLatencyMetrics = Field(default_factory=CallLatencyMetrics)


class ExecutionMetrics(_Artifact):
    """Privacy-safe aggregate request, repair, and per-model latency metrics."""

    logical_calls: int = Field(default=0, ge=0)
    wire_attempts: int = Field(default=0, ge=0)
    outer_retries: int = Field(default=0, ge=0)
    normal_chat_calls: int = Field(default=0, ge=0)
    ballot_repair_calls: int = Field(default=0, ge=0)
    ballots_repaired: int = Field(default=0, ge=0)
    unrecoverable_ballots: int = Field(default=0, ge=0)
    model_latency: dict[str, ModelLatencyMetrics] = Field(default_factory=dict)


class RunManifest(_Artifact):
    schema_version: str = RUN_MANIFEST_SCHEMA
    run_id: str
    resolved_config: dict[str, Any]
    input_provenance: dict[str, Any]
    task_count: int = Field(ge=0)
    artifact_schema_versions: dict[str, str]
    fingerprint: str
    checkpoint_compatibility: dict[str, Any]
    completion_status: Literal["complete", "incomplete"] = "incomplete"
    failed_tasks: tuple[FailedTask, ...] = ()
    case_seat_assignments: tuple[CaseSeatAssignment, ...] = ()
    execution_metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics)


def seat_id(index: int) -> str:
    """Return a pseudonymous stable seat identifier."""
    return f"seat-{index:02d}"


def _option_id(options: list[str], value: str | None) -> int | None:
    return options.index(value) if value in options else None


def project_artifacts(
    raw: dict[str, Any], *, run_id: str, task_index: int
) -> tuple[DetectorTrace, EvaluatorLabels]:
    """Project a private runner result into its public and restricted artifacts."""
    options = list(raw["options"])
    trace_id = f"{run_id}:q{task_index:04d}"
    seats = tuple(seat_id(int(agent_id)) for agent_id in sorted(raw["agents"], key=int))

    def decisions(name: str) -> tuple[Decision, ...]:
        values = raw[name]
        return tuple(
            Decision(
                seat_id=seat_id(int(agent_id)), option_id=_option_id(options, value.get("value"))
            )
            for agent_id, value in sorted(values.items(), key=lambda item: int(item[0]))
            if value is not None
        )

    public_rounds = tuple(
        PublicRound(
            round=round_data["round"],
            turns=tuple(
                PublicTurn(
                    seat_id=seat_id(turn["agent_id"]),
                    text=turn["text"],
                    decision_option_id=_option_id(options, turn.get("answer")),
                    observed_peer_turns=tuple(
                        PeerTurnReference(seat_id=seat_id(ref["agent_id"]), round=ref["round"] + 1)
                        for ref in turn["saw"]
                    ),
                )
                for turn in round_data["turns"]
            ),
        )
        for round_data in raw["rounds"]
    )
    public = DetectorTrace(
        trace_id=trace_id,
        question=raw["question"],
        options=tuple(Option(id=index, text=text) for index, text in enumerate(options)),
        participant_seat_ids=seats,
        initial_selected_option_ids=decisions("initial_decisions"),
        final_selected_option_ids=decisions("final_decisions"),
        rounds=public_rounds,
    )
    attack = raw["meta"]["attack"]
    labels = EvaluatorLabels(
        trace_id=trace_id,
        correct_option_id=options.index(raw["gold"]),
        attack=AttackLabels(
            present=attack is not None,
            mode=attack["mode"] if attack else None,
            placement=attack["placement"] if attack else None,
            adversary_seat_ids=tuple(seat_id(index) for index in attack["agent_ids"])
            if attack
            else (),
            cooperative_fact_holder_seat_ids=tuple(
                seat_id(int(item["agent_id"]))
                for item in raw["meta"]["fact_assignments"]
                if item["assigned_hidden_information"]
            ),
            target_option_id=_option_id(options, attack["target_answer"]) if attack else None,
            warmup_rounds=int(attack.get("warmup_rounds", 0)) if attack else 0,
        ),
        model_by_seat_id={seat_id(int(key)): value for key, value in raw["agents"].items()},
        benchmark_task_id=raw["task_id"],
        provenance={
            **dict(raw["meta"]["problem"]),
            "seat_assignment": dict(raw["meta"]["seat_assignment"]),
        },
        fact_assignments=tuple(raw["meta"]["fact_assignments"]),
        private_ballot_rationales={
            phase: {
                seat_id(int(agent_id)): decision["rationale"]
                for agent_id, decision in raw[phase].items()
                if decision is not None and decision.get("rationale")
            }
            for phase in ("initial_decisions", "final_decisions")
        },
    )
    return public, labels


def project_scratchpad(
    raw: dict[str, Any], *, run_id: str, task_index: int
) -> AttackerScratchpadLog | None:
    """Project a private runner result into its restricted scratchpad log.

    Returns ``None`` for a benign task: with no attack there is nothing to
    log, and no scratchpad file is written for that task.
    """
    attack = raw["meta"]["attack"]
    if attack is None:
        return None
    trace_id = f"{run_id}:q{task_index:04d}"
    entries = tuple(
        ScratchpadEntry(
            seat_id=seat_id(turn["agent_id"]),
            round=round_data["round"],
            goal=turn["scratchpad"].get("goal"),
            reasoning_steps=tuple(turn["scratchpad"].get("reasoning_steps") or ()),
            planned_action=turn["scratchpad"].get("planned_action"),
            raw_text=turn["scratchpad"]["raw_text"],
            parsed=turn["scratchpad"]["parsed"],
            discussion_text=turn["text"],
        )
        for round_data in raw["rounds"]
        for turn in round_data["turns"]
        if turn.get("scratchpad") is not None
    )
    return AttackerScratchpadLog(trace_id=trace_id, benchmark_task_id=raw["task_id"], entries=entries)
