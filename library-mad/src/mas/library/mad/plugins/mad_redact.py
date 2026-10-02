#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Keep an adversary's private planning out of its own next prompt.

``mad_turn`` decides what leaves a seat. This decides what comes back in.

The runtime commits the raw completion to the seat's history before any
pattern code runs, so without this the plan an adversary wrote in round 3
would be quoted back to it — and to the budget trimmer, and the summariser —
for every later round. A ``context_manager`` runs over committed history on
the way into each prompt, which is the supported place to drop it.

What it does NOT do: erase the raw string from the in-process history object
or from mas-lab's own debug events. Those are restricted artifacts. The
guarantee here is MAD's: the plan never reaches a peer, the seat's own future
context, or the public trace.
"""

from __future__ import annotations

from typing import Any

from mas.library.mad.domain.conversation import split_scratchpad
from mas.library.standard.plugins.context.conversation import StackConversation

__all__ = ["MadRedactConversation"]


class MadRedactConversation(StackConversation):
    """Full history (``StackConversation``), tagged private blocks stripped first."""

    def manage_history(
        self,
        past: list[dict[str, Any]],
        budget_tokens: int,
    ) -> list[dict[str, Any]]:
        redacted = [self._clean(message) for message in past or []]
        return super().manage_history(redacted, budget_tokens)

    @staticmethod
    def _clean(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("role") != "assistant":
            return message
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return message
        public, private = split_scratchpad(content)
        if private is None:
            return message
        return {**message, "content": public}
