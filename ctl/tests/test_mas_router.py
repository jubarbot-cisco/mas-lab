#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""MasAddressRouter — '@agent_id: text' parsed into RunningMas.ask()."""

from __future__ import annotations

from unittest.mock import MagicMock

from mas.ctl.session.mas_router import MasAddressRouter


def _router(*agent_ids: str):
    mas = MagicMock(agent_ids=frozenset(agent_ids))
    return MasAddressRouter(mas), mas


def test_unaddressed_turn_goes_to_the_entry_agent():
    router, mas = _router("entry", "peer")
    router.run_turn("@mention without a colon", auto_hitl=False)
    mas.ask.assert_called_once_with("@mention without a colon", auto_hitl=False)


def test_addressed_turn_names_the_agent():
    router, mas = _router("entry", "qa-agent.v2")
    router.run_turn("@qa-agent.v2: what's free Friday?", auto_hitl=False)
    mas.ask.assert_called_once_with("what's free Friday?", agent="qa-agent.v2", auto_hitl=False)


def test_unknown_agent_is_rejected_without_running_a_turn():
    router, mas = _router("entry")
    router.run_turn("@ghost: hello", auto_hitl=False)
    mas.ask.assert_not_called()
    error = mas.entry_controller.display.on_turn_error
    assert "unknown agent 'ghost'" in error.call_args[0][0]


def test_empty_message_to_a_known_agent_is_a_noop():
    router, mas = _router("entry", "peer")
    router.run_turn("@peer:   ", auto_hitl=False)
    mas.ask.assert_not_called()
    mas.entry_controller.display.on_turn_error.assert_called_once()


def test_hitl_resolution_routes_to_the_last_addressed_agent():
    router, mas = _router("entry", "peer")
    mas.active_agent_id = "peer"
    router.submit_hitl("resolve-object", auto_hitl=False)
    mas.controller.assert_called_once_with("peer")
    mas.controller.return_value.submit_hitl.assert_called_once_with(
        "resolve-object", auto_hitl=False
    )
