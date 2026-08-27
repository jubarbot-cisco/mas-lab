#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""Integration: '@agent_id: text' addresses a specific MAS agent end-to-end."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mas.ctl.executor import run_mas as run_mas_mod
from mas.ctl.executor.run_mas import execute_run_mas


def _write_two_agent_mas(tmp_path: Path) -> Path:
    agent_a = tmp_path / "agents" / "alpha.yaml"
    agent_b = tmp_path / "agents" / "beta.yaml"
    agent_a.parent.mkdir(parents=True)
    for path, name in ((agent_a, "alpha"), (agent_b, "beta")):
        path.write_text(
            f"""apiVersion: mas/v1
kind: Agent
metadata:
  name: {name}
spec:
  description: test agent
  models:
    - model: mock
""",
            encoding="utf-8",
        )

    mas_path = tmp_path / "mas.yaml"
    mas_path.write_text(
        """apiVersion: mas/v1
kind: MAS
metadata:
  name: two-agent-fixture
spec:
  agency:
    agents:
      - id: alpha
        ref: agents/alpha.yaml
      - id: beta
        ref: agents/beta.yaml
  workflow:
    entry: alpha
""",
        encoding="utf-8",
    )
    return mas_path


def test_addressed_query_reaches_the_named_agent_not_the_entry(tmp_path):
    """'@beta: ...' must build/run beta's own controller, not alpha's."""
    mas_path = _write_two_agent_mas(tmp_path)

    with patch.object(
        run_mas_mod,
        "build_agent_controller",
        wraps=run_mas_mod.build_agent_controller,
    ) as spy:
        rc = execute_run_mas(
            mas_path,
            queries=["hello alpha", "@beta: hello beta"],
            validate=False,
            infra_refs=["standard:mock-llm"],
            auto_hitl=True,
        )

    assert rc == 0
    assert [c.args[1] for c in spy.call_args_list] == ["beta"]
