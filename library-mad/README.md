# mas-library-mad

Multi-agent debate (MAD) attack-environment semantics, ported from
`mad_attack_env` onto mas-lab.

The mas-lab runtime is **not modified**. Everything here plugs in through the
normal extension points: a library manifest, two design-pattern plugins, and
one context-manager plugin.

## Layout

| Path | What it is |
|------|------------|
| `domain/` | MAD's pure logic, copied. Problems, deterministic setup, attack catalogue, prompts, response parsing, artifact schemas. |
| `compiler.py` | One MAD problem becomes one runnable MAS application. |
| `plugins/mad_debate.py` | The protocol: opening ballots, N frozen rounds, closing ballots. |
| `plugins/mad_turn.py` | One seat turn, published through an allow-list. |
| `plugins/mad_redact.py` | Keeps private planning out of a seat's later prompts. |
| `cli.py` | `mad-maslab compile` and `mad-maslab rerun`. |

### `domain/` is a copy, deliberately

The files under `domain/` are byte-identical to their upstream counterparts
except for rewritten imports, so `diff` against the generator stays readable:

```bash
diff mad_attack_env/debate/conversation.py \
     library-mad/src/mas/library/mad/domain/conversation.py
```

`tests/test_domain_parity.py` pins the behaviour that matters, so a silent
divergence fails rather than quietly changing the experiment.

### Built on mas-lab core, not beside it

The plugins are thin: `MadTurnPlugin` subclasses `SinglePassPlugin`
(`library-standard`), `MadRedactConversation` subclasses `StackConversation`,
and `mad_debate` dispatches through `dispatch_single` (same deterministic-
pattern helper every other mas-lab debate/voting pattern uses). Each only
overrides the one step MAD needs — parameter capture, public/private
sanitizing, or private-block stripping. Everything else — the turn state
machine, history management, delegation egress — is inherited.

## Replaying without a model

The native `llm_cache` infra middleware records every `LLM_CALL` to JSON and
can replay it without a provider — useful for debugging this port or for
cheap CI on a fixed cohort. See `docs/manifests/llm-cache.md`. Attach it via
`--infra-ref` on `rerun`; nothing in `library-mad` needs to change.

## Use

```bash
mad-maslab compile path/to/full58-fake-fact-single.yaml --out /tmp/hb-000 --index 0
mas-ctl validate /tmp/hb-000/mas.yaml /tmp/hb-000/agents
mad-maslab rerun /tmp/hb-000 --infra-ref team:llm-proxy
```

`compile` calls no model, so it is the cheap way to inspect exactly what each
seat is told before spending tokens.

Serving and interrogation are stock mas-lab:

```bash
mas-ctl serve-mas /tmp/hb-000/mas.yaml --token "$MAS_SERVE_TOKEN" \
  -q "Run the configured MAD debate."
```

## Where information is allowed to go

| Value | Lives in | Never in |
|-------|----------|----------|
| Private fact | that one seat's `spec.context` | MAS manifest, moderator, peers |
| Attack instruction | that adversary's `spec.context` | any other manifest |
| Public message | delegation result, debate result | — |
| Private planning | restricted sink (`domain/sink.py`) | peer prompts, seat's later prompts, debate result |
| Gold answer, attack metadata | `labels.json`, outside the manifests | anything the runtime loads |

Two enforcement points, because the runtime commits a raw completion to a
seat's history before pattern code runs:

* `mad_turn` controls what **leaves** a seat, so peers and the debate result
  only ever receive the public half.
* `mad_redact` controls what **re-enters** a seat, so an adversary's own plan
  is not quoted back to it in later rounds.

The raw completion still exists in the in-process history object and may
appear in mas-lab's own `events.jsonl`. Treat those as restricted artifacts.
MAD's three guarantees — not to a peer, not to the seat's future context, not
to the public trace — hold.

## Known divergences from MAD

Accepted deliberately; the generated dataset will not be byte-identical.

1. **No structured response schema.** mas-lab's engine does not send
   `response_format`, so turns use MAD's free-text path. Envelope compliance
   is therefore lower, and per-round `choice_index` falls back to
   `recover_decision`.
2. **Ballots enter history.** MAD elicits a ballot without mutating the
   agent's context; a mas-lab delegation always commits the turn.
3. **No ballot repair loop.** An unreadable ballot becomes a missing decision
   instead of two repair attempts.
4. **Sequential seats.** Seats are dispatched one at a time. Frozen-round
   visibility is unaffected; wall-clock and provider timing are.
5. **Sampling defaults.** MAD sends no temperature; mas-lab applies its own
   default unless the manifest sets one.
6. **Complete topology only.** MAD's chain/star/tree/random topologies
   (`debate/topology.py`) were not ported; `MadConfig` rejects any
   `protocol.topology` other than `"complete"`.
