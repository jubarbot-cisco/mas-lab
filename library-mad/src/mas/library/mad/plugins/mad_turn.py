#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""One debate seat: a single model turn, published through an allow-list.

MAD lets a turn carry two payloads — the message peers read, and, for an
adversary, the private plan behind it. This plugin is where they separate.
What it returns is what the moderator, the peers, and the public trace get;
the private half goes to the restricted sink instead.

Pairs with ``mad_redact`` (context_manager): this plugin controls what leaves
the seat, that one controls what comes back into the seat's next prompt.

Built on ``SinglePassPlugin``: the user->ctx->LLM_CALL->response state
machine is identical to one plain model turn, so only parameter capture and
the publish step are overridden here.
"""

from __future__ import annotations

from typing import Any

from mas.library.mad.domain import sink
from mas.library.mad.domain.conversation import parse_structured_turn
from mas.library.standard.plugins.design_patterns.single_pass import SinglePassPlugin
from mas.runtime.kernel.config import KernelConfig
from mas.runtime.kernel.response_text import response_text_from_run
from mas.runtime.kernel.state import QProduct, RunLedger
from mas.runtime.schema.egress import EgressSymbol, EmitClientResponse
from mas.runtime.schema.ingress import IngressSymbol, UserInputReceived


class MadTurnPlugin(SinglePassPlugin):
    """One LLM call per turn; publish the public half, file the private half."""

    plugin_id = "mad_turn@v1"

    def __init__(self) -> None:
        self._seat_id = ""
        self._attacker = False

    def handle_event(
        self,
        q: QProduct,
        run: RunLedger,
        event: IngressSymbol,
        *,
        config: KernelConfig,
    ) -> list[EgressSymbol]:
        if isinstance(event, UserInputReceived):
            params = _pattern_params(config)
            self._seat_id = str(params.get("seat_id") or "")
            self._attacker = bool(params.get("attacker"))
        return super().handle_event(q, run, event, config=config)

    def _evaluate(self, q: QProduct, run: RunLedger, config: KernelConfig) -> list[EgressSymbol]:
        """Split the completion, file the restricted half, return the public half.

        A ballot reply is not an envelope, so it falls through the same parser
        untouched and reaches the moderator verbatim — which is what the
        moderator's validator expects.
        """
        raw = response_text_from_run(run)
        parsed = parse_structured_turn(raw, attacker=self._attacker)
        sink.record(
            sink.TurnRecord(
                seat_id=self._seat_id,
                public_text=parsed.public_text,
                choice_index=parsed.choice_index,
                scratchpad=parsed.scratchpad.to_dict() if parsed.scratchpad else None,
                used_envelope=parsed.used_envelope,
            )
        )
        base = super()._evaluate(q, run, config)
        return [
            EmitClientResponse(content=parsed.public_text, finish_reason="stop")
            if isinstance(symbol, EmitClientResponse)
            else symbol
            for symbol in base
        ]


def _pattern_params(config: KernelConfig) -> dict[str, Any]:
    spec = getattr(config, "agent_spec", None) or {}
    if not isinstance(spec, dict):
        return {}
    binding = spec.get("design_pattern") or {}
    if not isinstance(binding, dict):
        return {}
    params = binding.get("params") or binding.get("config") or {}
    return params if isinstance(params, dict) else {}


__all__ = ["MadTurnPlugin"]
