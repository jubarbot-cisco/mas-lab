#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""MasAddressRouter — '@agent_id: text' addressing inside a running MAS."""

from __future__ import annotations

from unittest.mock import MagicMock

from mas.ctl.session.mas_router import _ADDRESS_RE, MasAddressRouter


class TestAddressRegex:
    def test_prefix_splits_agent_id_and_remainder(self):
        match = _ADDRESS_RE.match("@schedule_agent: what's free Friday?")
        assert match is not None
        assert match.groups() == ("schedule_agent", "what's free Friday?")

    def test_prefix_with_hyphenated_dotted_id(self):
        match = _ADDRESS_RE.match("@qa-agent.v2: ping")
        assert match is not None
        assert match.groups() == ("qa-agent.v2", "ping")

    def test_no_colon_is_not_a_prefix(self):
        assert _ADDRESS_RE.match("@mention this has no colon") is None


def _controller(agent_id: str) -> MagicMock:
    return MagicMock(agent_id=agent_id)


class TestMasAddressRouter:
    def test_unaddressed_turn_goes_to_default(self):
        default = _controller("entry")
        router = MasAddressRouter(
            default=default, agent_ids=frozenset({"entry", "peer"}), build_controller=MagicMock()
        )
        router.run_turn("hello", auto_hitl=False)
        default.run_turn.assert_called_once_with("hello", auto_hitl=False)

    def test_addressed_turn_builds_and_uses_peer_controller(self):
        default = _controller("entry")
        peer = _controller("peer")
        build = MagicMock(return_value=peer)
        router = MasAddressRouter(
            default=default, agent_ids=frozenset({"entry", "peer"}), build_controller=build
        )
        router.run_turn("@peer: hi there", auto_hitl=False)
        build.assert_called_once_with("peer")
        peer.run_turn.assert_called_once_with("hi there", auto_hitl=False)
        default.run_turn.assert_not_called()

    def test_peer_controller_is_cached_across_turns(self):
        default = _controller("entry")
        peer = _controller("peer")
        build = MagicMock(return_value=peer)
        router = MasAddressRouter(
            default=default, agent_ids=frozenset({"entry", "peer"}), build_controller=build
        )
        router.run_turn("@peer: first", auto_hitl=False)
        router.run_turn("@peer: second", auto_hitl=False)
        build.assert_called_once_with("peer")
        assert peer.run_turn.call_count == 2

    def test_unknown_agent_is_rejected_without_running_a_turn(self):
        default = _controller("entry")
        build = MagicMock()
        router = MasAddressRouter(
            default=default, agent_ids=frozenset({"entry"}), build_controller=build
        )
        router.run_turn("@ghost: hello", auto_hitl=False)
        build.assert_not_called()
        default.run_turn.assert_not_called()
        default.display.on_turn_error.assert_called_once()
        assert "unknown agent 'ghost'" in default.display.on_turn_error.call_args[0][0]

    def test_empty_message_to_agent_is_a_noop(self):
        default = _controller("entry")
        peer = _controller("peer")
        build = MagicMock(return_value=peer)
        router = MasAddressRouter(
            default=default, agent_ids=frozenset({"entry", "peer"}), build_controller=build
        )
        router.run_turn("@peer:   ", auto_hitl=False)
        build.assert_not_called()
        peer.run_turn.assert_not_called()
        default.display.on_turn_error.assert_called_once()

    def test_hitl_resolution_routes_to_last_addressed_controller(self):
        default = _controller("entry")
        peer = _controller("peer")
        build = MagicMock(return_value=peer)
        router = MasAddressRouter(
            default=default, agent_ids=frozenset({"entry", "peer"}), build_controller=build
        )
        router.run_turn("@peer: trigger hitl", auto_hitl=False)
        router.submit_hitl("resolve-object", auto_hitl=False)
        peer.submit_hitl.assert_called_once_with("resolve-object", auto_hitl=False)
        default.submit_hitl.assert_not_called()

    def test_default_controller_is_registered_under_its_own_agent_id(self):
        default = _controller("entry")
        router = MasAddressRouter(
            default=default, agent_ids=frozenset({"entry"}), build_controller=MagicMock()
        )
        # Addressing the entry agent explicitly should not build a second
        # controller for it — the default is reused.
        router.run_turn("@entry: hi", auto_hitl=False)
        default.run_turn.assert_called_once_with("hi", auto_hitl=False)
