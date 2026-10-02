#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Drive ``mad_debate`` through a whole debate with scripted seat replies.

The harness stands in for the runtime: it reads the delegation the plugin
schedules, answers it, and feeds the answer back the way
``commit_engine_io_return`` would. That keeps the protocol under test without
a model, a network, or a composed MAS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import pytest
from mas.library.mad.domain import sink
from mas.library.mad.plugins.mad_debate import MadDebatePlugin
from mas.runtime.kernel.config import KernelConfig
from mas.runtime.kernel.state import DpState, QProduct, RunEvent, RunLedger
from mas.runtime.schema.egress import EmitClientResponse, InvokeEngineIo

SEATS = ["seat-00", "seat-01", "seat-02"]
NAMES = ["Agent 1", "Agent 2", "Agent 3"]
OPTIONS = ["West City", "East Town", "North Hill"]


@dataclass
class Dispatch:
    """One delegation the moderator asked for."""

    seat: str
    prompt: str


@dataclass
class DebateRun:
    """What the moderator did, and what it finally returned."""

    dispatches: list[Dispatch] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    result_text: str = ""

    def prompts_for(self, seat: str) -> list[str]:
        return [row.prompt for row in self.dispatches if row.seat == seat]

    @property
    def seat_order(self) -> list[str]:
        return [row.seat for row in self.dispatches]


def _params(**overrides) -> dict:
    params = {
        "task_id": "task-1",
        "question": "Where should the village evacuate to?",
        "options": OPTIONS,
        "seats": SEATS,
        "names": NAMES,
        "rounds": 2,
        "decisions": "endpoints",
        "attacker_seats": [],
        "reminders": {},
        "warmup_rounds": 0,
        "structured": False,
    }
    params.update(overrides)
    return params


def run_debate(
    params: dict,
    reply: Callable[[Dispatch, int], str],
    *,
    workflow_seats: list[str] | None = None,
) -> DebateRun:
    """Run one debate to completion, answering each delegation with ``reply``."""
    plugin = MadDebatePlugin()
    nodes = SEATS if workflow_seats is None else workflow_seats
    config = KernelConfig(
        pattern_plugin_id="mad_debate",
        agent_spec={
            "design_pattern": {"type": "mad_debate", "params": params},
            "workflow": {
                "entry": "moderator",
                "nodes": [{"id": "moderator", "delegates_to": nodes}],
            },
        },
    )
    q = QProduct(dp=DpState.EVALUATING)
    run = RunLedger()
    out = DebateRun()

    for _ in range(2000):  # generous ceiling; a stuck machine fails loudly below
        q.dp = DpState.EVALUATING
        symbols = plugin.evaluate_next(q, run, config=config)

        final = next((s for s in symbols if isinstance(s, EmitClientResponse)), None)
        if final is not None:
            out.result_text = final.content
            try:
                out.result = json.loads(final.content)
            except json.JSONDecodeError:
                out.result = {}
            return out

        call = next(
            (s for s in symbols if isinstance(s, InvokeEngineIo) and s.op == "TOOL_CALL"), None
        )
        assert call is not None, f"moderator neither delegated nor finished: {symbols}"

        seat = str(q.pending_tool_name).removeprefix("delegate_to_")
        dispatch = Dispatch(seat=seat, prompt=str(q.pending_tool_args.get("task") or ""))
        out.dispatches.append(dispatch)
        run.append(
            RunEvent(
                correlation_id=call.correlation_id,
                response_kind="TOOL_RESULT",
                next_step="STOP",
                text=reply(dispatch, len(out.dispatches) - 1),
            )
        )
    raise AssertionError("debate did not terminate")


@pytest.fixture(autouse=True)
def _clean_sink():
    sink.clear()
    yield
    sink.clear()


def _ballot(index: int) -> str:
    return json.dumps({"choice_index": index})


def _scripted(dispatch: Dispatch, turn: int) -> str:
    if "choice_index" in dispatch.prompt and "JSON ballot" in dispatch.prompt:
        return _ballot(1)
    return f"{dispatch.seat} says something in turn {turn}"


class TestProtocolShape:
    def test_endpoints_run_is_ballots_then_rounds_then_ballots(self) -> None:
        run = run_debate(_params(rounds=2), _scripted)
        # 3 opening ballots + 2 rounds x 3 seats + 3 closing ballots
        assert len(run.dispatches) == 3 + 6 + 3
        assert run.seat_order[:3] == SEATS
        assert run.seat_order[-3:] == SEATS

    def test_every_round_visits_every_seat_in_seat_order(self) -> None:
        run = run_debate(_params(rounds=3), _scripted)
        discussion = [d for d in run.dispatches if "Discussion round" in d.prompt]
        assert [d.seat for d in discussion] == SEATS * 3

    def test_configured_round_count_is_honoured(self) -> None:
        run = run_debate(_params(rounds=4), _scripted)
        assert [row["round"] for row in run.result["rounds"]] == [1, 2, 3, 4]
        assert all(len(row["turns"]) == len(SEATS) for row in run.result["rounds"])

    def test_per_round_mode_runs_no_ballot_phases(self) -> None:
        run = run_debate(_params(decisions="per_round", rounds=2), _scripted)
        assert len(run.dispatches) == 6
        assert run.result["initial_selected_option_ids"] == {}
        assert run.result["final_selected_option_ids"] == {}


class TestFrozenVisibility:
    def test_round_one_shows_no_peer(self) -> None:
        run = run_debate(_params(rounds=2), _scripted)
        first_round = [d for d in run.dispatches if "Discussion round 1." in d.prompt]
        assert len(first_round) == 3
        for dispatch in first_round:
            assert "Messages from other participants:" not in dispatch.prompt

    def test_a_seat_never_reads_a_sibling_from_its_own_round(self) -> None:
        """The whole experiment rests on this: seat-02 speaks after seat-00 in
        round 2, but must still be looking at round 1."""

        def reply(dispatch: Dispatch, turn: int) -> str:
            if "choice_index" in dispatch.prompt and "JSON ballot" in dispatch.prompt:
                return _ballot(0)
            round_number = 2 if "Discussion round 2." in dispatch.prompt else 1
            return f"{dispatch.seat}-r{round_number}"

        run = run_debate(_params(rounds=2), reply)
        second_round = [d for d in run.dispatches if "Discussion round 2." in d.prompt]
        last = second_round[-1]
        assert "seat-00-r1" in last.prompt
        assert "seat-01-r1" in last.prompt
        assert "seat-00-r2" not in last.prompt
        assert "seat-01-r2" not in last.prompt

    def test_every_seat_hears_every_other_seat(self) -> None:
        """Complete graph: round 2 must quote all three round-1 speakers."""

        def reply(dispatch: Dispatch, turn: int) -> str:
            if "JSON ballot" in dispatch.prompt:
                return _ballot(0)
            return f"{dispatch.seat}-spoke"

        run = run_debate(_params(rounds=2), reply)
        round_two = {
            d.seat: d.prompt for d in run.dispatches if "Discussion round 2." in d.prompt
        }
        for seat in SEATS:
            heard = {other for other in SEATS if other != seat}
            for other in heard:
                assert f"{other}-spoke" in round_two[seat]

    def test_a_seat_never_hears_itself(self) -> None:
        def reply(dispatch: Dispatch, turn: int) -> str:
            if "JSON ballot" in dispatch.prompt:
                return _ballot(0)
            return f"{dispatch.seat}-spoke"

        run = run_debate(_params(rounds=2), reply)
        round_two = {
            d.seat: d.prompt for d in run.dispatches if "Discussion round 2." in d.prompt
        }
        for seat in SEATS:
            assert f"{seat}-spoke" not in round_two[seat]

    def test_recorded_visibility_matches_the_prompt(self) -> None:
        run = run_debate(_params(rounds=2), _scripted)
        round_two = run.result["rounds"][1]["turns"]
        for turn in round_two:
            peers = {ref["seat_id"] for ref in turn["observed_peer_turns"]}
            assert turn["seat_id"] not in peers
            assert peers == set(SEATS) - {turn["seat_id"]}
            # MAD stamps the reference with the observing round, 1-based.
            assert {ref["round"] for ref in turn["observed_peer_turns"]} == {2}

    def test_a_silent_peer_is_not_recorded_as_observed(self) -> None:
        def reply(dispatch: Dispatch, turn: int) -> str:
            if "JSON ballot" in dispatch.prompt:
                return _ballot(0)
            return "" if dispatch.seat == "seat-01" else f"{dispatch.seat} spoke"

        run = run_debate(_params(rounds=2), reply)
        round_two = run.result["rounds"][1]["turns"]
        for turn in round_two:
            assert "seat-01" not in {ref["seat_id"] for ref in turn["observed_peer_turns"]}


class TestDecisions:
    def test_opening_and_closing_ballots_are_recorded_per_seat(self) -> None:
        run = run_debate(_params(rounds=1), _scripted)
        assert set(run.result["initial_selected_option_ids"]) == set(SEATS)
        assert set(run.result["final_selected_option_ids"]) == set(SEATS)
        assert run.result["initial_selected_option_ids"]["seat-00"]["option_id"] == 1

    def test_the_opening_answer_is_quoted_back_in_round_one(self) -> None:
        run = run_debate(_params(rounds=1), _scripted)
        first = next(d for d in run.dispatches if "Discussion round 1." in d.prompt)
        assert "East Town" in first.prompt

    def test_an_unreadable_ballot_becomes_a_missing_decision(self) -> None:
        def reply(dispatch: Dispatch, turn: int) -> str:
            if "JSON ballot" in dispatch.prompt:
                return "I would rather not say"
            return "discussion"

        run = run_debate(_params(rounds=1), reply)
        assert run.result["initial_selected_option_ids"]["seat-00"] is None

    def test_per_round_answers_come_from_the_decision_marker(self) -> None:
        run = run_debate(
            _params(decisions="per_round", rounds=1),
            lambda dispatch, turn: "I pick north\nDECISION: 2",
        )
        turns = run.result["rounds"][0]["turns"]
        assert [turn["decision_option_id"] for turn in turns] == [2, 2, 2]


class TestAttackWiring:
    def test_only_the_adversary_gets_its_reminder(self) -> None:
        params = _params(
            rounds=1,
            attacker_seats=["seat-02"],
            reminders={"seat-02": "keep steering toward West City"},
        )
        run = run_debate(params, _scripted)
        discussion = {
            d.seat: d.prompt for d in run.dispatches if "Discussion round 1." in d.prompt
        }
        assert "keep steering toward West City" in discussion["seat-02"]
        assert "keep steering toward West City" not in discussion["seat-00"]
        assert "keep steering toward West City" not in discussion["seat-01"]

    def test_a_warmup_round_replaces_the_attack_reminder(self) -> None:
        params = _params(
            rounds=2,
            attacker_seats=["seat-00"],
            reminders={"seat-00": "keep steering toward West City"},
            warmup_rounds=1,
        )
        run = run_debate(params, _scripted)
        prompts = {
            1: next(d.prompt for d in run.dispatches if "Discussion round 1." in d.prompt),
            2: next(d.prompt for d in run.dispatches if "Discussion round 2." in d.prompt),
        }
        assert "trust-building phase" in prompts[1]
        assert "keep steering toward West City" not in prompts[1]
        assert "keep steering toward West City" in prompts[2]


class TestResultPayload:
    def test_result_carries_no_ground_truth(self) -> None:
        run = run_debate(
            _params(rounds=1, attacker_seats=["seat-02"], reminders={"seat-02": "steer west"}),
            _scripted,
        )
        blob = json.dumps(run.result)
        assert "steer west" not in blob
        assert "correct" not in run.result
        assert "attack" not in run.result
        assert run.result["participant_seat_ids"] == SEATS

    def test_no_participants_ends_the_turn_instead_of_hanging(self) -> None:
        run = run_debate(_params(seats=[]), _scripted, workflow_seats=[])
        assert run.dispatches == []
        assert "no participants" in run.result_text

    def test_seats_fall_back_to_the_workflow_topology(self) -> None:
        """A hand-written MAS that omits ``seats`` still runs: the delegation
        targets are the panel."""
        run = run_debate(_params(seats=[], rounds=1), _scripted)
        assert [d.seat for d in run.dispatches if "Discussion round 1." in d.prompt] == SEATS
