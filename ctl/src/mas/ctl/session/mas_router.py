#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""'@agent_id: text' addressing: the only place that knows stdin carries one
string per turn. Parses the prefix into ``RunningMas.ask(text, agent=...)``, and
duck-types SessionController so ``run_session_loop``/``OperatorConsole`` drive it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mas.runtime.driver.driver import DriverTrace

from mas.ctl.session.controller import TurnResult

if TYPE_CHECKING:  # executor imports session, never the reverse at runtime
    from mas.ctl.executor.running_mas import RunningMas

# Permissive on purpose: an id that matches but isn't materialized takes the
# "unknown agent" path below instead of silently parsing as plain text.
_ADDRESS_RE = re.compile(r"^@([A-Za-z0-9_.-]+):\s*(.*)$", re.DOTALL)


@dataclass
class MasAddressRouter:
    """Duck-types SessionController; routes '@agent_id:' turns to peers."""

    mas: RunningMas

    def __getattr__(self, name: str):
        return getattr(self.mas.entry_controller, name)

    @property
    def agent_ids(self) -> frozenset[str]:
        return self.mas.agent_ids

    @property
    def active_agent_id(self) -> str:
        return self.mas.active_agent_id

    def run_turn(self, text: str, **kwargs) -> TurnResult:
        match = _ADDRESS_RE.match(text)
        if match is None:
            return self.mas.ask(text, **kwargs)
        agent_id, remainder = match.groups()
        display = self.mas.entry_controller.display
        if agent_id not in self.mas.agent_ids:
            display.on_turn_error(
                f"unknown agent '{agent_id}' — known agents: "
                f"{', '.join(sorted(self.mas.agent_ids))}"
            )
            return TurnResult(trace=DriverTrace())
        if not remainder.strip():
            display.on_turn_error(f"no message for '{agent_id}' — nothing sent")
            return TurnResult(trace=DriverTrace())
        return self.mas.ask(remainder, agent=agent_id, **kwargs)

    def submit_hitl(self, resolve, **kwargs) -> TurnResult:
        return self.mas.controller(self.mas.active_agent_id).submit_hitl(resolve, **kwargs)
