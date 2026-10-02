#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""An adversary's private plan must not reach a peer or its own next prompt.

Two seams do that work, and both are tested here:

* ``mad_turn`` decides what leaves the seat.
* ``mad_redact`` decides what comes back in.

The sentinel below stands for the whole private payload: if it shows up
anywhere a peer or a later prompt can read, the experiment is compromised.
"""

from __future__ import annotations

import json

import pytest
from mas.library.mad.domain import sink
from mas.library.mad.plugins.mad_redact import MadRedactConversation
from mas.library.mad.plugins.mad_turn import MadTurnPlugin
from mas.runtime.kernel.config import KernelConfig
from mas.runtime.kernel.state import DpState, QProduct, RunEvent, RunLedger
from mas.runtime.schema.egress import EmitClientResponse

SENTINEL = "SECRET-PLAN-DO-NOT-LEAK"


@pytest.fixture(autouse=True)
def _clean_sink():
    sink.clear()
    yield
    sink.clear()


def _publish(raw: str, *, attacker: bool, seat: str = "seat-04") -> str:
    """Run one seat turn's publication step over ``raw`` and return what it emits."""
    plugin = MadTurnPlugin()
    config = KernelConfig(
        pattern_plugin_id="mad_turn",
        agent_spec={
            "design_pattern": {
                "type": "mad_turn",
                "params": {"seat_id": seat, "attacker": attacker},
            }
        },
    )
    plugin._seat_id = seat
    plugin._attacker = attacker
    q = QProduct(dp=DpState.EVALUATING)
    run = RunLedger()
    run.append(
        RunEvent(correlation_id=1, response_kind="MODEL_TEXT", next_step="STOP", text=raw)
    )
    symbols = plugin._evaluate(q, run, config)
    emitted = [s for s in symbols if isinstance(s, EmitClientResponse)]
    assert len(emitted) == 1
    return emitted[0].content


class TestWhatLeavesASeat:
    def test_envelope_private_half_is_not_published(self) -> None:
        raw = json.dumps(
            {
                "message": "The bridge looks passable to me.",
                "private": {
                    "goal": SENTINEL,
                    "reasoning_steps": [SENTINEL],
                    "planned_action": SENTINEL,
                },
            }
        )
        published = _publish(raw, attacker=True)
        assert published == "The bridge looks passable to me."
        assert SENTINEL not in published

    def test_tagged_private_block_is_not_published(self) -> None:
        raw = (
            f'<ATTACK_SCRATCHPAD>{{"goal": "{SENTINEL}", "reasoning_steps": [], '
            f'"planned_action": "{SENTINEL}"}}</ATTACK_SCRATCHPAD>\n\n'
            "The bridge looks passable to me."
        )
        published = _publish(raw, attacker=True)
        assert SENTINEL not in published
        assert "bridge looks passable" in published

    def test_unreadable_envelope_publishes_nothing_rather_than_the_plan(self) -> None:
        published = _publish(f'{{"message": "hi", "private": {{"goal": "{SENTINEL}"', attacker=True)
        assert published == ""

    def test_the_private_half_is_filed_for_the_evaluator(self) -> None:
        raw = json.dumps(
            {
                "message": "public point",
                "private": {
                    "goal": SENTINEL,
                    "reasoning_steps": ["step"],
                    "planned_action": "insist",
                },
            }
        )
        _publish(raw, attacker=True, seat="seat-04")
        record = sink.drain("seat-04")
        assert record is not None
        assert record.public_text == "public point"
        assert record.scratchpad is not None
        assert SENTINEL in json.dumps(record.scratchpad)

    def test_a_cooperative_seat_files_no_private_payload(self) -> None:
        _publish(json.dumps({"message": "public point"}), attacker=False, seat="seat-00")
        record = sink.drain("seat-00")
        assert record is not None and record.scratchpad is None

    def test_a_ballot_reaches_the_moderator_verbatim(self) -> None:
        assert _publish('{"choice_index": 2}', attacker=True) == '{"choice_index": 2}'

    def test_structured_choice_rides_the_sink_not_the_message(self) -> None:
        """Peers must not be handed the answer index just because the model
        returned it in the envelope."""
        raw = json.dumps({"message": "I favour the tunnel.", "choice_index": 1})
        published = _publish(raw, attacker=False, seat="seat-01")
        assert published == "I favour the tunnel."
        record = sink.drain("seat-01")
        assert record is not None and record.choice_index == 1


class TestWhatComesBackIn:
    def test_a_private_block_is_stripped_from_the_seats_own_history(self) -> None:
        """The runtime commits the raw completion before pattern code runs, so
        without this the plan is quoted back to its own author next round."""
        history = [
            {"role": "user", "content": "Discussion round 1."},
            {
                "role": "assistant",
                "content": (
                    f'<ATTACK_SCRATCHPAD>{{"goal": "{SENTINEL}"}}</ATTACK_SCRATCHPAD>\n\n'
                    "The bridge looks passable."
                ),
            },
        ]
        managed = MadRedactConversation().manage_history(history, 0)
        assert SENTINEL not in json.dumps(managed)
        assert "The bridge looks passable." in managed[1]["content"]

    def test_user_and_tool_messages_are_untouched(self) -> None:
        history = [
            {"role": "user", "content": "Discussion round 2."},
            {"role": "tool", "tool_call_id": "call_1", "content": "raw tool output"},
        ]
        assert MadRedactConversation().manage_history(history, 0) == history

    def test_clean_history_passes_through_unchanged(self) -> None:
        history = [
            {"role": "user", "content": "Discussion round 1."},
            {"role": "assistant", "content": "The bridge looks passable."},
        ]
        assert MadRedactConversation().manage_history(history, 0) == history

    def test_the_message_cap_still_applies(self) -> None:
        history = [{"role": "assistant", "content": f"m{i}"} for i in range(10)]
        managed = MadRedactConversation(max_messages=3).manage_history(history, 0)
        assert [m["content"] for m in managed] == ["m7", "m8", "m9"]

    def test_empty_history_is_safe(self) -> None:
        assert MadRedactConversation().manage_history([], 0) == []
