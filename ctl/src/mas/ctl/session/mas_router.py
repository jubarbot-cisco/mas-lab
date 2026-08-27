#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""'@agent_id: text' addressing for MAS sessions.

Routes one line of input to a specific agent materialized in a running MAS
instead of the entry agent, over the same SessionController-on-a-persistent-
instance mechanism delegation uses (``make_workflow_send``) and the same
MAS-wide ``session_id`` (``build_agent_controller``) — so addressing an agent
continues its real conversation instead of starting an isolated one.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from mas.runtime.driver.driver import DriverTrace

from mas.ctl.session.controller import SessionController, TurnResult

# Permissive on purpose: an id that matches but isn't materialized takes the
# "unknown agent" path below instead of silently parsing as plain text.
_ADDRESS_RE = re.compile(r"^@([A-Za-z0-9_.-]+):\s*(.*)$", re.DOTALL)


@dataclass
class MasAddressRouter:
    """Duck-types SessionController; routes '@agent_id:' turns to peers."""

    default: SessionController
    agent_ids: frozenset[str]
    build_controller: Callable[[str], SessionController]
    _controllers: dict[str, SessionController] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._controllers[self.default.agent_id] = self.default
        # Follow-up HITL lands on the agent that raised it, not always entry.
        self._active = self.default

    def __getattr__(self, name: str):
        return getattr(self.default, name)

    def run_turn(self, text: str, **kwargs) -> TurnResult:
        match = _ADDRESS_RE.match(text)
        if match is None:
            self._active = self.default
            return self.default.run_turn(text, **kwargs)
        agent_id, remainder = match.groups()
        if agent_id not in self.agent_ids:
            self.default.display.on_turn_error(
                f"unknown agent '{agent_id}' — known agents: "
                f"{', '.join(sorted(self.agent_ids))}"
            )
            return TurnResult(trace=DriverTrace())
        if not remainder.strip():
            self.default.display.on_turn_error(f"no message for '{agent_id}' — nothing sent")
            return TurnResult(trace=DriverTrace())
        controller = self._controllers.get(agent_id)
        if controller is None:
            controller = self.build_controller(agent_id)
            self._controllers[agent_id] = controller
        self._active = controller
        return controller.run_turn(remainder, **kwargs)

    def submit_hitl(self, resolve, **kwargs) -> TurnResult:
        return self._active.submit_hitl(resolve, **kwargs)
