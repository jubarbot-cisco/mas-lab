# MAD on mas-lab: the whole pipeline

How a HiddenBench or MMLU-Pro row becomes a running multi-agent debate on
mas-lab, and why that debate is the same experiment `mad_attack_env` runs.

Written for the team. It assumes you know mas-lab; it explains **our** wiring
and references core by `path:line` rather than re-explaining it.

```text
upstream dataset        HiddenBench 58  |  MMLU-Pro mmlu49_v2
        |
        v
MAD config YAML         protocol, panel, attack, llm
        |
        v
resolved setup          seats, facts, models, target, prompts   (domain/, pure)
        |
        v
MAS manifests           mas.yaml + agents/*.yaml + labels.json   (compiler.py)
        |
        v
mas-lab runtime         mad_debate + mad_turn + mad_redact       (plugins/)
        |
        v
artifacts               debate.json (public)  +  labels.json (restricted)
```

Two rules hold the whole design together:

1. **MAD owns the experiment.** What an agent is told, what it may see, which
   answer the adversary steers toward — all of it comes from code copied from
   `mad_attack_env`, not rewritten.
2. **mas-lab owns execution.** Composition, materialization, delegation,
   memory, serving. The runtime is not modified; everything plugs in through
   the normal library, design-pattern and context-manager extension points.

---

## 1. The two datasets

### HiddenBench — 58 four-fact scenarios

Prepared offline from a pinned upstream commit, not fetched at run time.

| Field | Value |
|---|---|
| Source | `HiddenBench_ICML` `data/benchmark.json` |
| Commit | `3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1` |
| Source tasks | 65 |
| Filter | `len(hidden_information) == 4` |
| Output tasks | **58** (7 excluded) |

The filter is what makes the panel size well-defined: four hidden facts means
four cooperative seats, one fact each. Provenance is recorded next to the data
in `benchmark_long_58.provenance.json`.

A scenario carries a description, `possible_answers`, `correct_answer`,
`shared_information` (everyone sees it) and `hidden_information` (split one
per seat).

### MMLU-Pro — the `mmlu49_v2` cohort

A 49-question low-consensus pool, sliced into three windows:

| Window | Pool rows | Used by |
|---|---|---|
| `benign-49.jsonl` | 0–48 | the benign control |
| `stealth-25.jsonl` | 0–24 | `v1-stealth`, `v11-stealth` |
| `colluding-24.jsonl` | 25–48 | `v1-colluding`, `v11-colluding` |

The two attacked windows are **disjoint and tile the benign window exactly
once**, so every attacked case has a benign control on the *same* question.

The property that makes the v1/v11 comparison controlled rather than merely
paired: `attack.target` is never set in these configs, so the target falls to
`select_target(options, correct, seed=...)`, and the task seed comes from
`protocol.seed` and the row index alone — never from the panel models or the
attack mode. A v1 arm and its v11 counterpart read the same window in the same
order and therefore resolve the **same wrong answer** on every question. The
adversary's prompt wording is the only free variable.

---

## 2. The MAD config file

One YAML describes one run. Annotated, using
`inputs/configs/hiddenbench/full58-fake-fact-single.yaml`:

```yaml
dataset:
  input: ../../data_qa/hiddenbench/benchmark_long_58.json
  n_samples: 58

protocol:
  facts: partitioned        # hidden facts split one per seat (vs `shared`)
  decisions: endpoints      # ballot before + after (vs `per_round`)
  rounds: 10
  topology: complete        # the port accepts only this; see section 9
  seed: 42

panel:
  models:                   # one per seat, in seat order
    - azure/gpt-4o-mini
    - vertex_ai/gemini-2.5-flash
    - azure/gpt-5-nano
    - bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0
    - azure/gpt-4o-mini     # the appended adversary
  roles: none               # no cooperative persona
  shuffle_roles: false

attack:
  mode: fake_fact_single    # invents one plausible false fact
  placement: additive       # appended seat, cooperative panel untouched
  seats: [4]
  warmup_rounds: 0          # attack starts in round 1

run:
  output_dir: output
  max_workers: 5
  name: hb-full58-fake-fact-single

llm:
  structured_output: best_effort
  request_timeout_seconds: 60
```

What each field actually drives:

| Field | Consequence |
|---|---|
| `protocol.facts` | `partitioned` ⇒ panel size = number of hidden facts. `shared` ⇒ panel size = `panel.n`. Also derives attack placement (section 3). |
| `protocol.decisions` | `endpoints` ⇒ an opening and a closing ballot phase exist. `per_round` ⇒ no ballots; each turn carries its own answer. |
| `protocol.rounds` | Discussion rounds. All nine target configs use 10. |
| `protocol.topology` | Who hears whom. The port implements `complete` only. |
| `protocol.seed` | With the row index, seeds every deterministic draw. |
| `panel.models` | One entry ⇒ broadcast to every seat. Otherwise the count **must** equal the resolved seat count, or setup raises. |
| `panel.roles` | `none` ⇒ no cooperative persona; the attack instruction takes that prompt slot. |
| `attack.mode` | Selects a row in the attack catalogue: instruction text, per-round reminder, seats required, whether it needs a target, whether it widens the adversary's view. |
| `attack.warmup_rounds` | Rounds the adversary spends earning trust first. During warm-up the mode's reminder is *replaced*, not merely preceded. |
| `llm.structured_output` | Changes the adversary's scratchpad directive wording, and (in MAD) attaches a JSON schema to the request. See section 9. |

---

## 3. Config + row → a resolved debate

This is the deterministic core, and it is MAD's code unchanged
(`domain/setup.py`, `domain/attacks.py`, `domain/conversation.py`). Given the
same config and the same row, it produces the same panel every time.

```text
problem row
    |
    v
_task_seed(seed, index) = seed + index * 10_000
    |
    +-- cooperative panel size
    |     partitioned -> len(hidden_facts)      (HiddenBench: 4)
    |     shared      -> panel.n                (MMLU: 6)
    |
    +-- attack placement        DERIVED, not configured:
    |     partitioned -> additive (adversary appended)
    |     shared      -> replace  (adversary takes an existing seat)
    |
    +-- _models(config, n)      one model id per seat
    |
    +-- build_views(...)        shuffle hidden facts, one per holder;
    |                           widens_view modes see all of them
    |
    +-- _seat_permutation(...)  stable shuffle when attack.shuffle_seats
    |
    +-- attacks.resolve(...)    target = select_target(options, correct, seed)
    |                           per-seat instruction + reminder, warm-up draw
    |
    v
build_system_prompt(...) per seat
```

A resolved system prompt is assembled in a fixed order:

1. the question,
2. for partitioned facts, a JSON `<FACTS>` block plus the
   information-asymmetry warning,
3. **either** a cooperative role **or** an attack instruction — never both
   (`build_system_prompt` raises if given both),
4. the shared instruction to argue from evidence and not count votes.

The gold answer never appears in any prompt.

### Attack modes in use

The catalogue holds 23 modes. The nine target configs use seven:

| Mode | Seats | Needs target | Widens view |
|---|---|---|---|
| `fake_fact_single` | 1 | yes | no |
| `fake_fact_colluding` | 2 | yes | no |
| `fact_diminisher` | 1 | yes | **yes** |
| `stealth` | 1 | yes | no |
| `stealth-v11` | 1 | yes | no |
| `tamas-colluding` | 2 | yes | no |
| `tamas-colluding-v11` | 2 | yes | no |

`fact_diminisher` is the only one that widens the adversary's view: it sees
every hidden fact, so it can downplay the evidence that points away from its
target. The others must invent or merely assert.

Additive attackers are validated to occupy exactly the appended seats — a
config that puts an additive adversary at seat 0 is rejected rather than
silently reshuffling the panel.

---

## 4. Conversion — `mad-maslab compile`

```bash
mad-maslab compile path/to/full58-fake-fact-single.yaml --out /tmp/hb-000 --index 0
```

No model is called. This is the cheap way to read exactly what each seat will
be told before spending tokens.

```text
/tmp/hb-000/
├── mas.yaml                 wiring only: who exists, who delegates to whom
├── agents/
│   ├── moderator.yaml       protocol params; no facts, no gold
│   ├── seat-00.yaml         one private fact, in spec.context
│   ├── seat-01.yaml
│   ├── seat-02.yaml
│   ├── seat-03.yaml
│   └── seat-04.yaml         the adversary: attack instruction, no private fact
└── labels.json              restricted: gold, target, fact map, resolved config
```

### Where each resolved value lands

| MAD value | Destination | Never in |
|---|---|---|
| Seat system prompt | that seat's `spec.context.mad_brief` | any other file |
| Seat model | that seat's `spec.models[0].model` | — |
| Adversary flag | that seat's `design_pattern.params.attacker` | — |
| Rounds, decisions, seats, names | moderator `design_pattern.params` | — |
| Per-seat attack reminders | moderator `design_pattern.params.reminders` | seat manifests |
| Gold answer | `labels.json` | every manifest |
| Attack mode, target, warm-up | `labels.json` | every manifest |
| Fact assignment map | `labels.json` | every manifest |

A seat manifest, abridged:

```yaml
apiVersion: mas/v1
kind: Agent
metadata:
  name: seat-04
spec:
  description: MAD debate participant seat-04.
  models:
    - model: azure/gpt-4o-mini
      max_tokens: 4000
  design_pattern:
    type: mad_turn
    params: {seat_id: seat-04, attacker: true}
  context_manager:
    type: mad_redact          # strips private planning on the way back in
  context:
    mad_brief: "<the MAD system prompt, verbatim>"
  working_memory:
    persistent: true
```

The moderator carries the protocol and nothing sensitive:

```yaml
  design_pattern:
    type: mad_debate
    params:
      task_id: "1"
      question: "..."
      options: [West City, East Town, North Hill]
      seats: [seat-00, seat-01, seat-02, seat-03, seat-04]
      names: ["Agent 1", ..., "Agent 5"]   # prompt-facing labels
      rounds: 10
      decisions: endpoints
      attacker_seats: [seat-04]
      reminders: {seat-04: "..."}
      warmup_rounds: 0
      structured: true
```

### Why one application per problem

Each problem has different private facts, a different target, sometimes a
different seat count. Those differences live in **seat system prompts**, and
mas-lab has no mechanism to vary those per dataset item:

- Dataset `memory_seeds` are applied with
  `apply_memory_seeds(instance, seeds)` — a **single instance**, the entry
  agent (`ctl/src/mas/ctl/benchmark/runner.py:600-601`). No per-agent
  targeting.
- `spec.params` is a flat dict staged on disk for **tools**, never merged into
  agent manifests (`ctl/src/mas/ctl/session/params_sidecar.py:18-43`).

So "one MAS template + 58 dataset items" cannot carry the experiment. One
compiled app per problem is not a workaround; it is the only placement that
keeps a private fact inside exactly one agent file.

`compile --count N` writes `q0000/`, `q0001/`, … one directory each.

---

## 5. Runtime — our wiring

```bash
mad-maslab rerun /tmp/hb-000 --infra-ref team:llm-proxy
```

`rerun` calls `build_running_mas(...)`, asks the entry agent one question
("Run the configured MAD debate."), then writes artifacts. Everything
interesting happens inside the three plugins.

### 5.1 `mad_debate` — the moderator

`plugins/mad_debate.py`. Subclasses `_DeterministicBase`, so it is a
deterministic pattern: it never asks a model what to do next.

State lives in `_MadState` (`:62`): phase, round index, seat cursor, the seat
it is waiting on, the frozen previous round, the current round, accumulated
rounds, opening and closing ballots.

Phases run `open → round × N → close → done`
(`_advance`, `:198`); the ballot phases are skipped entirely under
`per_round`.

```text
                    ┌──────────────────────────────────────────┐
                    │  freeze prev_public  (snapshot, once)     │
                    └────────────────────┬─────────────────────┘
                                         │
   for each seat in SEAT ORDER:          v
     build this seat's own prompt  ──> _round_prompt(:282)
       peers   = every other seat's frozen message   (_other_seats, :359)
       reminder= attacker ? attack_turn_reminder(...) : plain
       opening = own ballot answer, round 1 only
                                         │
                                         v
     dispatch_single(seat, prompt) ──> delegate_to_<seat> tool call
                                         │
                                         v
     result lands ──> _absorb(:143)
       record public text at this seat's index
       drain the restricted record from the sink
       record observed peers (1-based observing round, MAD's convention)
```

Two properties carry the experiment, and both are enforced here:

- **Frozen visibility.** `_close_round` (`:238`) copies the completed round
  into `prev_public` and clears the live buffer. Every prompt in round N is
  built from that snapshot, so a seat can never read a sibling that spoke
  earlier in its own round — even though seats run sequentially.
- **Seat order.** Seats are dispatched and recorded by index, so a turn is
  attributed to the agent that produced it regardless of timing.

Seats are dispatched **one at a time**. mas-lab's engine pool drains
sequentially anyway (`runtime/src/mas/runtime/engine/worker_pool.py:46`), so a
parallel fan-out would buy nothing today and would add a result-to-seat
alignment problem.

Ballots are validated by `parse_ballot` (`domain/ballots.py`), MAD's own
validator. An unreadable ballot becomes a missing decision rather than a
failed task.

### 5.2 `mad_turn` — one seat

`plugins/mad_turn.py`, 93 lines, most of them inherited. It subclasses
`SinglePassPlugin` from `library-standard` because one debate turn *is* one
plain model turn; only two things differ:

- `handle_event` (`:41`) captures `seat_id` and `attacker` from the agent spec
  on `UserInputReceived`, then defers to the parent.
- `_evaluate` (`:55`) splits the completion before anything else sees it.

```text
raw completion
      |
      v
parse_structured_turn(raw, attacker=...)        ← MAD's allow-list parser
      |
      +--> public_text   ──> EmitClientResponse ──> moderator, peers, trace
      |
      +--> choice_index  ──┐
      +--> scratchpad    ──┴─> sink.record(...)  ← restricted, never published
```

Publication is an **allow-list**: only `message` (and its ballot index) are
read out, so an unexpected field a model invents is withheld by construction
rather than needing to be recognised. A reply that looks like an envelope but
will not parse is withheld entirely — for an adversary, publishing it verbatim
would publish the whole plan.

A ballot reply is not an envelope, so it falls through untouched and reaches
the moderator's validator verbatim.

### 5.3 `mad_redact` — what comes back in

`plugins/mad_redact.py`, 50 lines. Subclasses `StackConversation`, so it
inherits history management and overrides only the strip.

This exists because the runtime commits the raw completion to the seat's
history *before* any pattern code runs
(`runtime/src/mas/runtime/driver/driver.py:630-631`). Without this, the plan an
adversary wrote in round 3 would be quoted back to it in round 4, and to the
budget trimmer, and to the summariser.

A `context_manager` runs over committed history on the way into every prompt,
which is the supported place to drop it.

### 5.4 The two enforcement points, together

| Direction | Component | Guarantees |
|---|---|---|
| What **leaves** a seat | `mad_turn` | peers, the moderator and `debate.json` only ever receive the public half |
| What **re-enters** a seat | `mad_redact` | the adversary's own plan is not quoted back to it later |

What this does **not** do: erase the raw string from the in-process history
object, or from mas-lab's own `events.jsonl`. Treat those as restricted
artifacts. MAD's three guarantees — not to a peer, not to the seat's future
context, not to the public trace — hold.

### 5.5 Call volume

Identical to MAD's own estimate
(`endpoints ⇒ rounds + 2 ballots per seat`, `per_round ⇒ rounds per seat`):

| Cohort | Seats | Rounds | Calls per case |
|---|---|---|---|
| HiddenBench benign | 4 | 10 | 4 × 12 = **48** |
| HiddenBench single / diminisher | 5 | 10 | 5 × 12 = **60** |
| HiddenBench colluding | 6 | 10 | 6 × 12 = **72** |
| MMLU (per_round) | 6 | 10 | 6 × 10 = **60** |

---

## 6. Artifacts

### `artifacts/debate.json` — public

Emitted by the moderator (`_result`, `mad_debate.py:325`). Contains nothing
restricted — no gold, no attack metadata, no scratchpads — so it is safe to
hand to a detector *once projected into the canonical envelope* (section 10).

```json
{
  "schema_version": "mad-maslab-debate/v1",
  "task_id": "1",
  "question": "...",
  "options": ["West City", "East Town", "North Hill"],
  "participant_seat_ids": ["seat-00", "...", "seat-04"],
  "initial_selected_option_ids": {"seat-00": {"option_id": 1, "rationale": "..."}},
  "final_selected_option_ids":   {"seat-00": {"option_id": 1, "rationale": "..."}},
  "rounds": [
    {"round": 1, "turns": [
      {"seat_id": "seat-00",
       "text": "...",
       "decision_option_id": null,
       "observed_peer_turns": [{"seat_id": "seat-01", "round": 1}],
       "used_envelope": true}
    ]}
  ]
}
```

Mapped against MAD's `mad-public-trace/v1`:

| MAD | Ours | Note |
|---|---|---|
| `trace_id` | `task_id` | MAD mints `run:qNNNN`; ours is per-app, so the directory is the identity |
| `question` | same | |
| `options` | same name | ours is a plain list of strings, MAD uses `{id, text}` objects |
| `participant_seat_ids` | same | identical `seat-NN` pseudonyms |
| `initial`/`final_selected_option_ids` | same | MAD uses `{seat_id, option_id}` rows; ours is a map and keeps the ballot rationale inline |
| `rounds[].turns[].text` | same | the published half only |
| `observed_peer_turns` | same | same 1-based observing-round convention |

> **Not drop-in for the canonical loader.** The detection repo's
> `shared/trace_loader.py` validates with `set(trace) != required` *and*
> `schema_version != "mad-public-trace/v1"` — an exact field-set match. Our
> file fails both checks. The *information* is equivalent and complete; the
> *envelope* is not. Converting is a small, purely mechanical projection; see
> section 10.

### `labels.json` — restricted

Written by the compiler (`_labels`, `compiler.py:186`), outside the manifests
entirely:

```json
{
  "schema_version": "mad-maslab-labels/v1",
  "benchmark_task_id": "1",
  "correct_option_id": 1,
  "model_by_seat_id": {"seat-00": "azure/gpt-4o-mini", "...": "..."},
  "attack": {"present": true, "mode": "fake_fact_single", "placement": "additive",
             "adversary_seat_ids": ["seat-04"], "target_option_id": 2,
             "warmup_rounds": 0},
  "fact_assignments": [...],
  "seat_assignment": {...},
  "resolved_config": {...}
}
```

Equivalent to MAD's `mad-evaluator-labels/v1`, plus the full resolved config so
a run can be reproduced from the sidecar alone.

---

## 7. Serving and interrogation

Stock mas-lab, nothing of ours involved:

```bash
mas-ctl serve-mas /tmp/hb-000/mas.yaml --token "$MAS_SERVE_TOKEN" \
  -q "Run the configured MAD debate."
```

The debate runs as the bootstrap query, then every seat stays addressable over
REST and synchronous A2A (`GET /agents`, `POST /agents/{id}/`). Each answers
from its own controller and retained history, so you can ask seat-04 what it
was doing — and it will answer from the public half of its own conversation,
because that is all `mad_turn` ever committed.

---

## 8. Why this is on par with `mad_attack_env`

Three tiers, strongest first.

### Tier 1 — identical by construction

The experiment is not reimplemented; it is the same functions. `domain/` is a
copy, and the copy is checked:

| File | Status |
|---|---|
| `problems.py`, `attacks.py`, `artifacts.py`, `json_extract.py` | **byte-identical** to source |
| `content.yaml` | **byte-identical** |
| `conversation.py` | import rewrites only |
| `config.py`, `setup.py` | import rewrites + the complete-only change (section 9) |

Verify at any time:

```bash
diff mad_attack_env/debate/attacks.py \
     mas-lab/library-mad/src/mas/library/mad/domain/attacks.py
```

Because `conversation.py` is untouched, **the prompts are byte-identical to
MAD's** — system prompts, round prompts, attack reminders, warm-up wording,
ballot prompts, peer rendering. Including the structured-output directive: the
`structured` flag flows config → moderator params → `attack_turn_reminder`,
exactly as MAD passes it.

That covers: problem normalization, panel sizing, fact assignment, model
assignment, seat permutation, attack selection, target selection, warm-up
draw, prompt assembly, response parsing, decision recovery, artifact schemas.

### Tier 2 — structurally equivalent, and pinned by tests

The round loop was rewritten (imperative loop → plugin state machine), so it
is held in place by property tests rather than by inspection:

| MAD property | Enforced in | Pinned by |
|---|---|---|
| Round 1 sees no peer | `_round_prompt` | `test_round_one_shows_no_peer` |
| A seat never reads a same-round sibling | `_close_round` snapshot | `test_a_seat_never_reads_a_sibling_from_its_own_round` |
| Complete graph: all peers visible | `_other_seats` | `test_every_seat_hears_every_other_seat` |
| No self-loops | `_other_seats` | `test_a_seat_never_hears_itself` |
| Silent peers are not recorded | `_absorb` | `test_a_silent_peer_is_not_recorded_as_observed` |
| Seat order independent of timing | sequential dispatch | `test_every_round_visits_every_seat_in_seat_order` |
| Ballots bracket the discussion | `_advance` | `test_endpoints_run_is_ballots_then_rounds_then_ballots` |
| Opening answer quoted in round 1 | `_round_prompt` | `test_the_opening_answer_is_quoted_back_in_round_one` |
| Only the adversary gets its reminder | `_round_prompt` | `test_only_the_adversary_gets_its_reminder` |
| Warm-up replaces the reminder | `attack_turn_reminder` | `test_a_warmup_round_replaces_the_attack_reminder` |
| Private plan never published | `mad_turn._evaluate` | `TestWhatLeavesASeat` (7 tests) |
| Private plan never re-enters | `mad_redact` | `TestWhatComesBackIn` (5 tests) |
| Gold never enters a manifest | `compiler` | `test_the_gold_answer_never_enters_a_manifest` |
| A hidden fact reaches one seat | `compiler` | `test_a_hidden_fact_reaches_exactly_one_seat` |

### Tier 3 — identical quantities

Same seat count, same seat identities (`seat-NN`), same model per seat, same
resolved target, same per-case call count (section 5.5), same prompt-facing
agent names. Compiling twice byte-matches
(`test_compiling_twice_produces_identical_files`).

### What "on par" means precisely

**The experimental design is the same**, and the public trace carries the same
information with the same semantics — every field a detector reads is present
and means the same thing.

**The artifact envelope is not the same.** `debate.json` is not accepted by the
detection repo's canonical loader as-is; it needs a mechanical projection into
`mad-public-trace/v1` (section 10). That is packaging, not experiment, but it
is real work that stands between this pipeline and the existing detectors.

**It is not byte-reproducible.** Model sampling is non-deterministic, and the
divergences in section 9 shift the distribution slightly. Two runs of *MAD
itself* are not byte-identical either; the claim is equivalence of the
experiment, not of the output.

---

## 9. Where it is not on par

Accepted deliberately. Listed with honest impact.

| # | Divergence | Impact |
|---|---|---|
| 1 | **`response_format` not sent.** Prompts are identical — the adversary is still *asked* for the JSON envelope — but mas-lab's engine does not attach the schema, so nothing *enforces* it. | Envelope compliance becomes model-dependent. MAD recorded 8.5–70% scratchpad coverage on the free-text path versus ~100% under schema. Affects the restricted channel, not the public trace. |
| 2 | **Ballots mutate history.** MAD elicits a ballot without touching the agent's context; a mas-lab delegation always commits the turn. | The opening ballot appears in round-1 context. MAD already feeds the opening answer into the round-1 prompt, so this is mild duplication rather than new information. The closing ballot is last, so it affects nothing. |
| 3 | **No ballot repair loop.** MAD retries a malformed ballot twice. | More missing decisions under models that return prose instead of JSON. Deliberate: a wrong answer is worse than no answer. |
| 4 | **Sampling defaults.** MAD sends no temperature; mas-lab applies its own default unless the manifest sets one. | Different sampling distribution. Set `temperature` in the seat manifest to pin it. |
| 5 | **Sequential seats.** | Wall-clock and provider timing only. Frozen-round visibility is unaffected by construction. |
| 6 | **Complete topology only.** | None in practice — all nine target configs use `complete`. A non-complete config is rejected rather than silently run. |

Item 1 is the one to watch if you care about scratchpad-based analysis. Items
2–6 do not change what a detector sees.

---

## 10. Known gaps

### Artifacts are not in the canonical detector envelope

`debate.json` carries everything `mad-public-trace/v1` carries, but under a
different schema name and field set, so
`ioc-governance-security-detections-techniques/shared/trace_loader.py` rejects
it — that loader demands an exact field-set match plus
`schema_version == "mad-public-trace/v1"`.

This is the one gap that blocks reuse of the existing detectors.

Closing it is mechanical: emit `trace_id` as `<run>:qNNNN`, render `options`
as `{id, text}` objects, reshape the decision maps into `{seat_id, option_id}`
rows, and move the ballot rationale into the labels sidecar where MAD keeps it
(`private_ballot_rationales`). `domain/artifacts.py` is already a byte-identical
copy of MAD's projector, so the clean version is a shim that adapts our
moderator result into the dict shape `project_artifacts()` expects, then calls
it — giving `mad-public-trace/v1` and `mad-evaluator-labels/v1` for free.

### Scratchpads are parsed but not persisted

`mad_turn` extracts the adversary's private plan and files it in the sink;
`_absorb` (`mad_debate.py:143`) drains that record but reads only
`choice_index` and `used_envelope`. The scratchpad itself is dropped.

- **Privacy is unaffected** — the plan is still removed from the public text,
  peer prompts and the seat's own later context. That is tested.
- **The artifact is missing** — MAD writes
  `scratchpads/<run>/qNNNN.json`; we write nothing equivalent.

Treated as a debugging aid rather than a required output, so it is deferred.
Closing it is small: carry `restricted.scratchpad` into the round record and
have `rerun` write `artifacts/scratchpads.json`. Two docstrings
(`cli.py:13`, `mad_debate.py:329`) currently describe the intended behaviour
rather than the actual one and should be corrected in the same change.

### Batch execution

`compile --count N` writes N applications, but `rerun` takes one at a time.
MAD's cross-task concurrency, fingerprinted checkpoint and resume are not
ported. When that lands, reuse `JsonCheckpointStore` from ctl rather than
re-porting MAD's `_Checkpoint`.

---

## 11. Verification

```bash
# package tests — 68, all green
.venv/bin/pytest library-mad/tests -q

# lint
.venv/bin/ruff check library-mad

# full repo unit gate — 1485 passed, 2 skipped
.venv/bin/pytest tests/ runtime/tests/ ctl/tests/ \
  lab/components/core/tests/ lab/components/bench/tests/ \
  library-standard/tests/ library-mad/tests/ -q

# the core is untouched — both must print nothing
git status --porcelain=v1 -- runtime/ ctl/src/
```

Test files:

| File | Covers |
|---|---|
| `test_domain_parity.py` | the copied MAD behaviour: determinism, prompts, decisions, the private split |
| `test_mad_debate.py` | the protocol: phases, frozen visibility, seat order, ballots, attack wiring |
| `test_private_output.py` | both privacy seams, with a leak sentinel |
| `test_compiler.py` | manifest shape and the information boundaries |

`test_mad_debate.py` drives the plugin through a whole debate with scripted
seat replies — no model, no network, no composed MAS — by reading the
delegation it schedules and feeding results back the way
`commit_engine_io_return` would.

---

## 12. End to end

```bash
# 1. compile one problem (offline, no model calls)
mad-maslab compile \
  mad_attack_env/inputs/configs/hiddenbench/full58-fake-fact-single.yaml \
  --out /tmp/hb-000 --index 0

# 2. check what each seat will be told
cat /tmp/hb-000/agents/seat-04.yaml      # the adversary's brief
cat /tmp/hb-000/labels.json              # gold + target, outside the manifests

# 3. validate the manifests
mas-ctl validate /tmp/hb-000/mas.yaml /tmp/hb-000/agents

# 4. dry run against the mock provider (no spend, no network)
mad-maslab rerun /tmp/hb-000 --infra-ref standard:mock-llm

# 5. the real thing
mad-maslab rerun /tmp/hb-000 --infra-ref team:llm-proxy

# 6. or keep it open and interrogate the seats
mas-ctl serve-mas /tmp/hb-000/mas.yaml --token "$MAS_SERVE_TOKEN" \
  -q "Run the configured MAD debate."
```

For a cheap deterministic replay, attach the native `llm_cache` middleware via
`--infra-ref` (see `docs/manifests/llm-cache.md`); nothing in `library-mad`
needs to change.
