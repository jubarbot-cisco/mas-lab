# Serving agents over HTTP

`mas-ctl run-mas` drives a MAS from stdin. `mas-ctl serve-mas` holds the same MAS
open and lets anything on the network address its agents individually — each
answers from its own controller and the shared MAS session, so it can be asked
about what it already did.

The motivating case is an external agent interrogating the agents of a running
MAS to judge their behaviour, but the surface is general: any client that can
speak HTTP can reach any agent by id.

Distinct from `mas-lab serve`, which starts the lab controller daemon and runs a
MAS as short-lived subprocess jobs.

## Start a server

```bash
pip install 'mas-ctl[serve]'
mas-ctl serve-mas mas.yaml --token "$MAS_SERVE_TOKEN" --port 8080
```

Composition flags match `run-mas`: `--overlay`, `--deployment`, `--flavour`,
`--infra-ref`, `--kernel`, `--no-validate`, plus the observability and trace
flags. Server flags: `--host` (default `127.0.0.1`), `--port` (default `8080`),
`--token`, `--no-auth`.

One MAS per process. Scaling is more processes.

## Bootstrapping with a query

`-q/--query` runs turns before the server starts:

```bash
mas-ctl serve-mas mas.yaml --no-auth \
  -q "plan a trip to Rome" \
  -q "@beta: how many stops?"
```

Repeat `-q` for a multi-turn opening, and use `@agent_id: text` to aim a turn at
one agent — same addressing as the interactive CLI.

Served turns reuse the same controllers and session, so an agent asked over HTTP
answers with the bootstrap conversation in its history.

## Routes

| Route | Purpose |
|-------|---------|
| `GET /agents` | List agents with their endpoint and card URLs |
| `POST /agents/{id}/ask` | Plain REST: `{"text": "..."}` |
| `GET /agents/{id}/.well-known/agent-card.json` | A2A agent card |
| `POST /agents/{id}/` | A2A JSON-RPC `message/send` |
| `POST /mcp` | MCP JSON-RPC: `initialize`, `tools/list`, `tools/call` |

REST serves local tooling and `curl`. A2A lets a foreign agent talk to ours with
no client code written here. `GET /agents` is a deliberate non-A2A addition:
A2A has no standard directory for a multi-agent host, so a client needs some way
to enumerate targets.

Agent ids are the ids from the MAS manifest — the same names the interactive CLI
addresses with `@agent_id: text`.

```bash
curl -s -H "Authorization: Bearer $MAS_SERVE_TOKEN" localhost:8080/agents

curl -s -X POST -H "Authorization: Bearer $MAS_SERVE_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"text":"what did you do earlier?"}' \
  localhost:8080/agents/planner/ask
```

```json
{"agent": "planner", "status": "ok", "text": "...", "error": null}
```

## A2A

Implemented: the agent card and `message/send`. Tasks complete synchronously and
`contextId` is the MAS session id.

```bash
curl -s -X POST -H "Authorization: Bearer $MAS_SERVE_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":"1","method":"message/send",
       "params":{"message":{"role":"user",
                 "parts":[{"kind":"text","text":"identify yourself"}]}}}' \
  localhost:8080/agents/alpha/
```

Not implemented: `message/stream`, `tasks/get`, `tasks/cancel`, push
notifications. Cards advertise `"streaming": false` so clients do not attempt
them.

## MCP

A card says which tools an agent *claims*. `POST /mcp` lets you call one and see
what it actually returns, so a client can fact-check an answer against the tool
that produced it.

```bash
curl -s -X POST -H "Authorization: Bearer $MAS_SERVE_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":"1","method":"tools/call",
       "params":{"name":"calc","arguments":{"expression":"2 + 40"}}}' \
  localhost:8080/mcp
```

The tool set is flat and global, built once at startup from the agents' live
tool providers. A tool declared by several agents is one entry listing all of
them in `declared_by`; providers are per-agent but the tools are stateless, so
the instances are interchangeable. Two agents declaring the same *name* with
different contracts is disambiguated as `name@agent_id` — nothing is dropped and
nothing is fatal, so every tool of every agent stays reachable.

Calls run in the live MAS, on the same instances the agents use, under the
declaring agent's lock. A value from a fresh instance would not be the value the
agent saw. Results come back as text, exactly as the agent's own tool loop sees
them. A tool that raises returns `isError: true` rather than a transport error —
a failing tool is evidence, not an outage.

Stateless: `initialize` returns no session id, and any sent is ignored.

Cards are built once at startup by re-reading each agent's manifest, and describe
what an agent **declares** — role, tools, delegation peers — never what it did.
A client comparing the declared role against actual replies is comparing claim
against behaviour, which is the point.

## Authentication

Every route requires `Authorization: Bearer <token>`. The token comes from
`--token` or `$MAS_SERVE_TOKEN`, and one is required unless you pass `--no-auth`
explicitly. When auth is on, cards advertise the scheme so an A2A client can
discover the requirement.

The server binds `127.0.0.1` by default; exposing it to other hosts is a
deliberate `--host 0.0.0.0`.

## Turn semantics

Served turns use the agent's live controller and the MAS-wide session, so an
agent answers with its full history — and the question enters that history too.
Turns are recorded in `traces/events.jsonl` exactly like any other, with no
marker distinguishing HTTP-driven turns from the rest.

Turn-level outcomes always return HTTP 200 with the result in the body; a silent
or failing agent is an observation, not an exception.

| Condition | REST | A2A |
|-----------|------|-----|
| Turn ok | 200 `status: "ok"` | Task `completed` |
| Turn error or raised | 200 `status: "error"` | Task `failed` |
| Awaiting HITL | 200 `status: "awaiting_hitl"` | Task `failed` |
| Unknown agent | 404 | JSON-RPC `-32001` |
| Malformed body | 422 | JSON-RPC `-32700` / `-32602` |
| Missing or bad token | 401 | 401 |
| Text starting with `/` | 400 | JSON-RPC `-32602` |

Text beginning with `/` is rejected rather than forwarded: `run_turn` intercepts
`/steer`, and a remote client gets to ask, never to steer. Operator console verbs
(`/reset`, `/pause`, `/resume`, `/abort`) are likewise not exposed — `ask` is the
only verb.

A turn that raises is reported in the body and the server keeps serving. One
poisoned turn must not take down the surface for every other agent.

## Concurrency

`SessionController` is synchronous and not thread-safe, so turns run in a
threadpool under one lock per agent id. Turns to the same agent serialize — one
conversation advances one turn at a time — while turns to different agents run in
parallel.
