#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Ballot parsing for the moderator.

MAD validates an endpoint ballot inside its own elicitation loop. The port
splits those apart: mas-lab performs the call, the moderator validates what
came back. This is the same validator, exposed under a public name.
"""

from __future__ import annotations

from mas.library.mad.domain.conversation import Decision, _parse_ballot

__all__ = ["parse_ballot"]


def parse_ballot(raw: str, options: list[str]) -> Decision | None:
    """Return the validated ballot, or None when it cannot be read.

    MAD retries a malformed ballot twice before giving up. The port has no
    repair loop yet, so an unreadable ballot becomes a missing decision rather
    than a failed task — a wrong answer would be worse than no answer.
    """
    try:
        return _parse_ballot(raw or "", options)
    except ValueError:
        return None
