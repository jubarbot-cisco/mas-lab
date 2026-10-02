#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Restricted side channel between a debate seat and its moderator.

A seat returns one thing to the moderator: the text its peers are allowed to
read. Everything else a turn produces — the adversary's private plan, the
structured ballot index, whether the model honoured the response envelope —
is ground truth about the run, not conversation, and must not travel through
a prompt, a delegation result, or the seat's own history.

So it travels here instead: the seat's plugin appends a record, the moderator
pops it in the same order it dispatched. One queue per seat, FIFO, because
the moderator drives seats one at a time and reads each result before moving
on.

ponytail: in-process dict, single run per process. A seat interrogated out of
band (an A2A question mid-debate) would enqueue a record nobody pops; drop the
queue between runs rather than reconciling it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = ["TurnRecord", "clear", "drain", "pending", "record"]


@dataclass(frozen=True)
class TurnRecord:
    """What one seat produced, minus what its peers may see."""

    seat_id: str
    public_text: str
    choice_index: int | None = None
    scratchpad: dict[str, Any] | None = None
    used_envelope: bool = True


_QUEUES: dict[str, deque[TurnRecord]] = {}


def record(entry: TurnRecord) -> None:
    """Append one seat's restricted turn payload."""
    _QUEUES.setdefault(entry.seat_id, deque()).append(entry)


def drain(seat_id: str) -> TurnRecord | None:
    """Pop the oldest unread record for ``seat_id``, or None when empty."""
    queue = _QUEUES.get(seat_id)
    if not queue:
        return None
    return queue.popleft()


def pending(seat_id: str) -> int:
    """Return how many records are queued for ``seat_id``."""
    return len(_QUEUES.get(seat_id) or ())


def clear(seat_id: str | None = None) -> None:
    """Drop one seat's queue, or every queue when ``seat_id`` is None."""
    if seat_id is None:
        _QUEUES.clear()
        return
    _QUEUES.pop(seat_id, None)
