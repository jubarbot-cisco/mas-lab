#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Run ``mad_attack_env`` debates as live mas-lab applications.

``mas.library.mad.domain`` carries the experiment (ported from MAD, pure).
``mas.library.mad.compiler`` turns one problem into manifests.
``mas.library.mad.plugins`` executes the protocol on the mas-lab runtime.

The runtime itself is untouched: everything here plugs in through the normal
library, design-pattern and context-manager extension points.
"""

from mas.library.mad.compiler import CompiledApp, compile_problem  # noqa: F401
