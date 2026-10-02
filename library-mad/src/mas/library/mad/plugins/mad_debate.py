#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""The MAD protocol as a deterministic pattern: ballots, rounds, visibility.

MAD's runner is an imperative loop that owns every phase of one debate. Here
the same phases are plugin state, and each seat's turn is a delegation to a
materialized agent instead of a direct model call. The parts that define the
experiment are unchanged and come from ``mad.domain``: the prompts, the
attack reminders, the ballot validator.

This port implements only MAD's complete graph: every seat hears every
other seat's previous-round message, and no seat hears itself. (MAD also
supports chain/star/tree/random topologies via an adjacency matrix; none of
the configs this port targets use them, so that module was not carried
over. Restore ``domain/topology.py`` from ``mad_attack_env`` if one is ever
needed.)

Two properties carry the experiment and are enforced here:

* **Frozen visibility.** Every prompt in round N is built from a snapshot of
  round N-1 taken before the round starts. A seat can never read a sibling
  that spoke earlier in its own round.
* **Seat order.** Seats are dispatched and recorded in seat order, so a turn
  is attributed to the agent that produced it regardless of timing.

ponytail: seats are dispatched one at a time. mas-lab's engine pool drains
sequentially anyway, so a parallel fan-out would buy nothing today and would
add a result-to-seat alignment problem. Revisit if the pool becomes real.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mas.library.mad.domain import sink
from mas.library.mad.domain.ballots import parse_ballot
from mas.library.mad.domain.config import DebateDecisions
from mas.library.mad.domain.conversation import (
    attack_turn_reminder,
    ballot_prompt,
    build_utterance_prompt,
    recover_decision,
    render_peers,
)
from mas.library.standard.plugins.design_patterns.base import _DeterministicBase, _TurnState
from mas.library.standard.plugins.design_patterns.utils import dispatch_single
from mas.runtime.kernel.config import KernelConfig
from mas.runtime.kernel.state import QProduct, RunLedger
from mas.runtime.schema.egress import EgressSymbol

RESULT_SCHEMA = "mad-maslab-debate/v1"

_PHASE_OPEN = "open"
_PHASE_ROUND = "round"
_PHASE_CLOSE = "close"
_PHASE_DONE = "done"


@dataclass
class _MadState(_TurnState):
    """One debate in flight."""

    phase: str = _PHASE_OPEN
    round_index: int = 0
    seat_cursor: int = 0
    awaiting_seat: int = -1
    seen_results: int = 0
    prev_public: list[str] = field(default_factory=list)
    current_public: list[str] = field(default_factory=list)
    current_turns: list[dict[str, Any]] = field(default_factory=list)
    rounds_out: list[dict[str, Any]] = field(default_factory=list)
    opening: dict[str, Any] = field(default_factory=dict)
    closing: dict[str, Any] = field(default_factory=dict)
    started: bool = False


class MadDebatePlugin(_DeterministicBase):
    """Opening ballots, N frozen rounds, closing ballots, then one result."""

    plugin_id = "mad_debate@v1"
    mode = "mad_debate"

    def _make_state(self) -> _TurnState:
        return _MadState()

    def _mad_state(self) -> _MadState:
        return self._state()  # type: ignore[return-value]

    # --- configuration ----------------------------------------------------

    @staticmethod
    def _params(config: KernelConfig) -> dict[str, Any]:
        spec = getattr(config, "agent_spec", None) or {}
        if not isinstance(spec, dict):
            return {}
        binding = spec.get("design_pattern") or {}
        if not isinstance(binding, dict):
            return {}
        params = binding.get("params") or binding.get("config") or {}
        return params if isinstance(params, dict) else {}

    def _seats(self, config: KernelConfig) -> list[str]:
        """Seat agent ids, in seat order.

        Falls back to the workflow's delegation targets so the pattern still
        runs on a hand-written MAS, but a compiled app always states them.
        """
        seats = self._params(config).get("seats")
        if isinstance(seats, list) and seats:
            return [str(seat) for seat in seats]
        return self._participants_from_spec(config)

    # --- main loop --------------------------------------------------------

    def evaluate_next(
        self, q: QProduct, run: RunLedger, *, config: KernelConfig
    ) -> list[EgressSymbol]:
        params = self._params(config)
        seats = self._seats(config)
        if not seats:
            return self._no_participants(q)

        state = self._mad_state()
        self._start(state, params, seats)
        self._absorb(state, run, params, seats)
        return self._advance(q, run, config, state, params, seats)

    def _start(self, state: _MadState, params: dict[str, Any], seats: list[str]) -> None:
        if state.started:
            return
        state.started = True
        state.phase = _PHASE_OPEN if _endpoints(params) else _PHASE_ROUND
        state.prev_public = [""] * len(seats)
        state.current_public = [""] * len(seats)

    @staticmethod
    def _results(run: RunLedger) -> list[str]:
        """Every delegation result so far, empty ones kept so seats stay aligned."""
        return [event.text for event in run.events if event.response_kind == "TOOL_RESULT"]

    def _absorb(
        self, state: _MadState, run: RunLedger, params: dict[str, Any], seats: list[str]
    ) -> None:
        """Record the seat we were waiting on, if its result just landed."""
        results = self._results(run)
        if len(results) <= state.seen_results or state.awaiting_seat < 0:
            return
        state.seen_results = len(results)
        seat_index = state.awaiting_seat
        state.awaiting_seat = -1
        text = results[-1]
        seat_id = seats[seat_index]
        restricted = sink.drain(seat_id)

        if state.phase in (_PHASE_OPEN, _PHASE_CLOSE):
            decision = parse_ballot(text, list(params.get("options") or []))
            target = state.opening if state.phase == _PHASE_OPEN else state.closing
            target[seat_id] = (
                None
                if decision is None
                else {"option_id": decision.index, "rationale": decision.rationale}
            )
            return

        state.current_public[seat_index] = text
        state.current_turns.append(
            {
                "seat_id": seat_id,
                "text": text,
                "decision_option_id": self._choice(text, restricted, params),
                "observed_peer_turns": [
                    # MAD labels a reference with the round the observation
                    # happened in (1-based), not the round the peer spoke in.
                    {"seat_id": seats[peer], "round": state.round_index + 1}
                    for peer in _other_seats(seats, seat_index)
                    if state.round_index > 0 and state.prev_public[peer]
                ],
                "used_envelope": bool(restricted.used_envelope) if restricted else False,
            }
        )

    @staticmethod
    def _choice(
        text: str, restricted: sink.TurnRecord | None, params: dict[str, Any]
    ) -> int | None:
        """Per-round answer: the structured index when present, else the marker."""
        if not _per_round(params):
            return None
        options = list(params.get("options") or [])
        if restricted is not None and restricted.choice_index is not None:
            if 0 <= restricted.choice_index < len(options):
                return restricted.choice_index
        decision = recover_decision(text, options)
        return None if decision is None else decision.index

    def _advance(
        self,
        q: QProduct,
        run: RunLedger,
        config: KernelConfig,
        state: _MadState,
        params: dict[str, Any],
        seats: list[str],
    ) -> list[EgressSymbol]:
        rounds = int(params.get("rounds") or 1)

        while True:
            if state.phase == _PHASE_OPEN:
                if state.seat_cursor < len(seats):
                    return self._dispatch(q, run, config, state, seats, self._ballot(params, True))
                state.phase = _PHASE_ROUND
                state.seat_cursor = 0
                continue

            if state.phase == _PHASE_ROUND:
                if state.seat_cursor < len(seats):
                    prompt = self._round_prompt(state, params, seats, state.seat_cursor)
                    return self._dispatch(q, run, config, state, seats, prompt)
                self._close_round(state)
                if state.round_index < rounds:
                    state.seat_cursor = 0
                    continue
                state.phase = _PHASE_CLOSE if _endpoints(params) else _PHASE_DONE
                state.seat_cursor = 0
                continue

            if state.phase == _PHASE_CLOSE:
                if state.seat_cursor < len(seats):
                    prompt = self._closing_prompt(state, params)
                    return self._dispatch(q, run, config, state, seats, prompt)
                state.phase = _PHASE_DONE
                continue

            return self._finish(q, self._result(state, params, seats))

    def _close_round(self, state: _MadState) -> None:
        """Seal the round, then freeze it as the only thing the next one sees."""
        state.rounds_out.append(
            {"round": state.round_index + 1, "turns": list(state.current_turns)}
        )
        state.prev_public = list(state.current_public)
        state.current_public = [""] * len(state.current_public)
        state.current_turns = []
        state.round_index += 1

    def _dispatch(
        self,
        q: QProduct,
        run: RunLedger,
        config: KernelConfig,
        state: _MadState,
        seats: list[str],
        prompt: str,
    ) -> list[EgressSymbol]:
        seat_index = state.seat_cursor
        state.awaiting_seat = seat_index
        state.seat_cursor += 1
        return dispatch_single(self, q, run, config, prompt, seats[seat_index])

    # --- prompts ----------------------------------------------------------

    @staticmethod
    def _ballot(params: dict[str, Any], opening: bool) -> str:
        return ballot_prompt(list(params.get("options") or []), opening=opening)

    def _closing_prompt(self, state: _MadState, params: dict[str, Any]) -> str:
        question = self._ballot(params, False)
        if not state.rounds_out:
            return question
        names = list(params.get("names") or [])
        final = state.rounds_out[-1]["turns"]
        transcript = render_peers(
            [
                (_name(names, _seat_index(params, turn["seat_id"])), turn["text"])
                for turn in final
            ]
        )
        return f"The discussion is complete.\n\n{transcript}\n\n{question}"

    def _round_prompt(
        self, state: _MadState, params: dict[str, Any], seats: list[str], seat_index: int
    ) -> str:
        names = list(params.get("names") or [])
        options = list(params.get("options") or [])
        seat_id = seats[seat_index]

        peers: list[tuple[str, str]] = []
        if state.round_index > 0:
            for peer in _other_seats(seats, seat_index):
                utterance = state.prev_public[peer]
                if utterance:
                    peers.append((_name(names, peer), utterance))

        reminders = params.get("reminders") or {}
        attackers = set(params.get("attacker_seats") or [])
        if seat_id in attackers:
            reminder = attack_turn_reminder(
                str(reminders.get(seat_id) or ""),
                structured=bool(params.get("structured")),
                round_number=state.round_index + 1,
                warmup_rounds=int(params.get("warmup_rounds") or 0),
            )
        else:
            reminder = reminders.get(seat_id) or None

        opening = state.opening.get(seat_id) if state.round_index == 0 else None
        own_opening = None
        if opening and opening.get("option_id") is not None:
            own_opening = options[int(opening["option_id"])]

        return build_utterance_prompt(
            round_index=state.round_index,
            peers=peers,
            options=options,
            decisions=_decisions(params),
            own_opening=own_opening,
            attack_reminder=reminder,
        )

    # --- result -----------------------------------------------------------

    @staticmethod
    def _result(state: _MadState, params: dict[str, Any], seats: list[str]) -> str:
        """The debate as public record: seats, rounds, what each seat could see.

        Ground truth is not here. Gold, attack metadata and fact assignments
        stay in the compiler's restricted sidecar, and scratchpads stay in the
        sink, so this payload is safe to hand to a detector.
        """
        return json.dumps(
            {
                "schema_version": RESULT_SCHEMA,
                "task_id": params.get("task_id") or "",
                "question": params.get("question") or "",
                "options": list(params.get("options") or []),
                "participant_seat_ids": list(seats),
                "initial_selected_option_ids": dict(state.opening),
                "final_selected_option_ids": dict(state.closing),
                "rounds": list(state.rounds_out),
            },
            ensure_ascii=False,
        )


def _decisions(params: dict[str, Any]) -> DebateDecisions:
    return DebateDecisions(str(params.get("decisions") or DebateDecisions.ENDPOINTS.value))


def _endpoints(params: dict[str, Any]) -> bool:
    return _decisions(params) is DebateDecisions.ENDPOINTS


def _per_round(params: dict[str, Any]) -> bool:
    return _decisions(params) is DebateDecisions.PER_ROUND


def _other_seats(seats: list[str], seat_index: int) -> list[int]:
    """Every seat index except ``seat_index`` — the complete-graph audience."""
    return [index for index in range(len(seats)) if index != seat_index]


def _name(names: list[str], seat: int) -> str:
    if 0 <= seat < len(names):
        return str(names[seat])
    return f"Agent {seat + 1}"


def _seat_index(params: dict[str, Any], seat_id: str) -> int:
    seats = [str(seat) for seat in (params.get("seats") or [])]
    return seats.index(seat_id) if seat_id in seats else 0


__all__ = ["MadDebatePlugin", "RESULT_SCHEMA"]
