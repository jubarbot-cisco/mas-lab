#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""MAD experiment semantics, ported from ``mad_attack_env``.

These modules are copies of the upstream pure logic — problem normalization,
deterministic setup, attack catalogue, prompts, response parsing. They are
kept byte-identical to their source except for import rewrites (and, in
``config``/``setup``, dropping MAD's non-complete topologies — see below),
so ``diff`` against the generator stays readable and drift stays visible.

Nothing in here may reach for the network, a model, or a runtime object: the
experiment is data and pure functions, execution belongs to mas-lab.

Source modules:
    common/json_extract.py  -> json_extract.py
    debate/problems.py      -> problems.py
    debate/attacks.py       -> attacks.py
    debate/config.py        -> config.py
    debate/conversation.py  -> conversation.py
    debate/setup.py         -> setup.py
    debate/artifacts.py     -> artifacts.py
    inputs/content.yaml     -> content.yaml

``json_extract``, ``problems``, ``attacks``, ``artifacts`` and
``content.yaml`` are byte-identical to their source. ``conversation``
differs only in import lines.

``config`` and ``setup`` have real divergences beyond imports. ``setup``
packages ``content.yaml`` under this module instead of a top-level
``inputs`` package. Both files also implement only MAD's complete topology
(every seat hears every other seat): ``debate/topology.py`` was not carried
over, ``config`` rejects any other ``protocol.topology``, and ``setup`` no
longer builds or permutes an adjacency matrix. Restore
``debate/topology.py`` from ``mad_attack_env`` and reapply it to both files
if another topology is ever needed.
"""

from __future__ import annotations

__all__ = ["MAD_SOURCE_MODULES"]

MAD_SOURCE_MODULES = (
    "common/json_extract.py",
    "debate/problems.py",
    "debate/attacks.py",
    "debate/config.py",
    "debate/conversation.py",
    "debate/setup.py",
    "debate/artifacts.py",
)
