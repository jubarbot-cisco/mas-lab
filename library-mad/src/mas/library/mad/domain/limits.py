#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Size ceilings shared by the copied MAD domain modules.

Upstream these live on the MAD LLM client, which is not ported: mas-lab owns
transport. Only the parsing ceiling travels with the parser.
"""

from __future__ import annotations

MAX_MODEL_RESPONSE_CHARS = 100_000

__all__ = ["MAX_MODEL_RESPONSE_CHARS"]
