#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""MCP surface: the tools an agent claims, invoked for real. Failure paths first."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fastapi.testclient import TestClient
from mas.ctl.serve.app import build_app
from mas.ctl.serve.mcp import ToolEntry, build_tools

TOKEN = "s3cret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def rpc(client, method, params=None, headers=AUTH):
    body = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", json=body, headers=headers)


@dataclass
class _StubProvider:
    """Duck-types ManifestToolProvider for the failure paths."""

    specs: list = field(default_factory=list)
    raises: bool = False

    def list_tools(self):
        return self.specs

    def call_tool(self, tool_name, arguments, *, ctx=None, user=""):
        if self.raises:
            raise RuntimeError("boom")
        return {"tool": tool_name, "args": arguments}


class _StubMas:
    session_id = "sess-1"

    @property
    def agent_ids(self):
        return frozenset({"alpha"})

    def ask(self, text, *, agent=None, **kwargs):  # pragma: no cover - unused here
        raise AssertionError("no turns in these tests")


CALC_SPEC = {
    "name": "calc",
    "description": "Add numbers.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
}
SCHEMA = CALC_SPEC["parameters"]


def _client(provider):
    entry = ToolEntry("calc", "Add numbers.", SCHEMA, ["alpha"], "alpha", provider)
    return TestClient(build_app(_StubMas(), {}, token=TOKEN, tools={"calc": entry}))


def test_auth_rejects_the_mcp_endpoint():
    client = _client(_StubProvider())
    assert client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).status_code == 401
    assert rpc(client, "tools/list").status_code == 200


def test_unknown_tool_is_a_protocol_error():
    body = rpc(_client(_StubProvider()), "tools/call", {"name": "nope"}).json()
    assert body["error"]["code"] == -32601
    assert "nope" in body["error"]["message"]


def test_non_object_arguments_are_rejected():
    body = rpc(_client(_StubProvider()), "tools/call", {"name": "calc", "arguments": [1]}).json()
    assert body["error"]["code"] == -32602


def test_unsupported_method_is_rejected():
    assert rpc(_client(_StubProvider()), "resources/list").json()["error"]["code"] == -32601


def test_a_raising_tool_is_a_finding_not_a_transport_fault():
    """The detective must record 'the tool failed', not see a 500."""
    resp = rpc(_client(_StubProvider(raises=True)), "tools/call", {"name": "calc"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["isError"] is True
    assert "boom" in result["content"][0]["text"]


def test_malformed_request_is_a_parse_error():
    client = _client(_StubProvider())
    body = client.post("/mcp", json={"method": "tools/list"}, headers=AUTH).json()
    assert body["error"]["code"] == -32700


# ---- build_tools ---------------------------------------------------------


class _FakeMas:
    """Enough of RunningMas for build_tools: agent_ids + tool_provider()."""

    def __init__(self, providers: dict):
        self._providers = providers

    @property
    def agent_ids(self):
        return frozenset(self._providers)

    def tool_provider(self, agent_id):
        return self._providers.get(agent_id)


def test_a_tool_shared_by_two_agents_collapses_to_one_entry():
    """trip-planner really does this — query_graph_database is on three agents."""
    tools = build_tools(
        _FakeMas({"alpha": _StubProvider([CALC_SPEC]), "beta": _StubProvider([CALC_SPEC])})
    )
    assert list(tools) == ["calc"]
    assert tools["calc"].declared_by == ["alpha", "beta"]


def test_same_name_different_contract_stays_reachable_under_both_agents():
    """Nothing is dropped and nothing is fatal: every tool of every agent is callable."""
    other = dict(CALC_SPEC, description="Something else entirely.")
    tools = build_tools(
        _FakeMas({"alpha": _StubProvider([CALC_SPEC]), "beta": _StubProvider([other])})
    )
    assert sorted(tools) == ["calc", "calc@beta"]


# ---- end to end ----------------------------------------------------------


def _write_mas(tmp_path: Path) -> Path:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "adder.py").write_text(
        '''from mas.runtime.contracts import ToolContract


class AdderTool(ToolContract):
    def on_collect_tools(self, **_):
        return [
            {
                "name": "adder",
                "description": "Add two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            }
        ]

    def on_execute_tool(self, tool_name, arguments, **_):
        return {"sum": int(arguments["a"]) + int(arguments["b"])}
''',
        encoding="utf-8",
    )
    (tmp_path / "agents" / "adder.tool.yaml").write_text(
        """apiVersion: mas/v1
kind: Tool
metadata:
  name: adder
  description: Add two numbers.
spec:
  description: Add two numbers.
  parameters:
    - name: a
      type: integer
      required: true
    - name: b
      type: integer
      required: true
  impl:
    kind: python
    module_path: ./adder.py
    class_name: AdderTool
""",
        encoding="utf-8",
    )
    for name in ("alpha", "beta"):
        (tmp_path / "agents" / f"{name}.yaml").write_text(
            f"""apiVersion: mas/v1
kind: Agent
metadata:
  name: {name}
spec:
  description: test agent
  tools:
    - ref: ./adder.tool.yaml
  context:
    role: "You compute."
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
  name: mcp-fixture
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


def test_a_real_tool_runs_in_the_live_mas(tmp_path):
    """The point of the feature: call the agents' own tool and get its real value."""
    from mas.ctl.executor.run_mas import build_running_mas

    mas = build_running_mas(
        _write_mas(tmp_path), validate=False, infra_refs=["standard:mock-llm"], single_turn=True
    )
    try:
        tools = build_tools(mas)
        assert "adder" in tools
        assert tools["adder"].declared_by == ["alpha", "beta"]  # shared, one entry

        client = TestClient(build_app(mas, {}, token=None, tools=tools))
        listed = rpc(client, "tools/list", headers=None).json()["result"]["tools"]
        assert [t["name"] for t in listed] == ["adder"]
        assert listed[0]["inputSchema"]["properties"]

        body = rpc(
            client,
            "tools/call",
            {"name": "adder", "arguments": {"a": 2, "b": 40}},
            headers=None,
        ).json()
        assert body["result"]["isError"] is False
        assert "42" in body["result"]["content"][0]["text"]
    finally:
        mas.close()
