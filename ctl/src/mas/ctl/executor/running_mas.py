#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""A materialized MAS held open, so something other than stdin can drive it.

List its agents, ``ask`` one by id (``agent=None`` -> entry agent), ``close`` it.
Every agent shares the MAS-wide ``session_id``, so addressing one continues its
real conversation and lands in the same events sink as the rest of the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mas.ctl.executor.mas_session import MaterializedMas, build_agent_controller
from mas.ctl.session.controller import SessionController, TurnResult, close_observability


@dataclass
class RunningMas:
    materialized: MaterializedMas
    entry_agent_id: str
    session_id: str
    entry_controller: SessionController
    verbose: int = 0
    plugin_set: Any | None = None
    scoped_recorders: tuple[Any, ...] = ()
    # Last agent addressed — a HITL follow-up belongs to whoever raised it,
    # which is not always the entry agent.
    active_agent_id: str = ""
    _controllers: dict[str, SessionController] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._controllers[self.entry_agent_id] = self.entry_controller
        self.active_agent_id = self.entry_agent_id

    @property
    def agent_ids(self) -> frozenset[str]:
        return frozenset(self.materialized.materialized.instances)

    def controller(self, agent_id: str) -> SessionController:
        controller = self._controllers.get(agent_id)
        if controller is None:
            controller = build_agent_controller(
                self.materialized, agent_id, session_id=self.session_id, verbose=self.verbose
            )
            self._controllers[agent_id] = controller
        return controller

    def ask(self, text: str, *, agent: str | None = None, **kwargs) -> TurnResult:
        """One turn to ``agent`` (default: entry). KeyError if not materialized."""
        controller = self.controller(agent or self.entry_agent_id)
        self.active_agent_id = controller.agent_id
        return controller.run_turn(text, **kwargs)

    def close(self) -> None:
        close_observability(self.entry_controller)
        for recorder in self.scoped_recorders:
            recorder.close()
