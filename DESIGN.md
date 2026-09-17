# System One Harness — Design

A harness (orchestration layer) that gives **System One decision models** (TypeSafe's
"System One" model class, e.g. Jev) the three abilities they lack natively:

1. **iterate** — run a multi-turn loop instead of a single shot,
2. **call tools** — execute side-effecting actions under model control,
3. **access external datasources** — read DBs, HTTP APIs, files, memory, etc.

---

## 1. The core constraint

A System One model is a **decision function**, not a text generator:

```
(state: unstructured|structured, questions: {name: typed question}) -> decisions
```

- It emits typed answers only: **Choice** (pick from a fixed option set), **Score**
  (rate on an ordered rubric), **Noul** (yes/no with calibrated probability).
- Every answer carries a **calibrated probability**. **Choice** and **Score** add a
  separate **confidence**; **Noul** returns only a probability (no separate confidence
  field), so this harness reuses the probability as the confidence — an approximation,
  and a less principled gating signal than Choice/Score confidence.
- It evaluates all questions **in parallel against one shared state** — questions in a
  single call cannot reference each other's answers.
- It **cannot** generate strings, so it cannot write tool-call arguments, prose, or
  self-directed instructions.
- It **cannot** self-loop or perform I/O.

Consequence: the model is a **policy** `π(state, questions) → P(action)`. Everything with
side effects, memory, or time lives in the harness. This is not a workaround — it matches
TypeSafe's intended usage: their docs describe "confidence-gated routing" and
"speculative fan-out", and prescribe score-then-choose composition for large option sets
(the pattern §6 calls composite scoring).

> **LLM contrast.** An LLM agent generates the next text (including tool-call JSON) and
> reasons via chain-of-thought. A System One harness pre-declares the finite action space
> and lets the model emit a *calibrated choice* over it. The model never produces an
> out-of-schema value, and always reports how sure it is.

---

## 2. Architecture

```
                     ┌──────────────────────────────────────────────────────────┐
                     │                        RUNNER (the loop)                  │
                     │                                                          │
   task ──────────►  │   State ──► StateBuilder.assemble() ──► state text        │
                     │      ▲                                      │             │
                     │      │                          ┌───────────▼──────────┐  │
                     │      │                          │  System One client    │  │
                     │      │                          │  evaluate(state, Qs)  │  │
                     │      │                          └───────────┬──────────┘  │
                     │      │                                      │ decisions   │
                     │      │                          ┌───────────▼──────────┐  │
                     │      │                          │  ConfidenceGate       │  │
                     │      │                          │  act/confirm/escalate │  │
                     │      │                          └───────────┬──────────┘  │
                     │      │                                      │             │
                     │      │                          ┌───────────▼──────────┐  │
                     │      │                          │  Policy.resolve()     │  │
                     │      │                          │  decisions → actions  │  │
                     │      │                          └───────────┬──────────┘  │
                     │      │                                      │             │
                     │      │                 ┌────────────────────▼──────┐      │
                     │      │                 │  ToolRegistry.execute()   │      │
                     │      │                 │  (idempotent, budgeted)   │      │
                     │      │                 └────────────────────┬──────┘      │
                     │      └────────────── results / events ───────┘             │
                     │                                                           │
                     │   ContextProviders (DB/HTTP/files/memory) re-gather each   │
                     │   turn, so the state is refreshed before every evaluate.   │
                     └───────────────────────────────────────────────────────────┘
```

### Components

| Component | Responsibility |
|---|---|
| `State` | The working memory: task, structured fields, decision history, tool-event log. |
| `StateBuilder` | Renders `State` + external context into the bounded text/blob the model consumes. |
| `SystemOneClient` | Thin adapter over the decision API (TypeSafe `jev-latest` or a pinned version, or a mock). |
| `Policy` | Declares the **decision space**: which questions to ask, how answers map to `Action`s. |
| `ConfidenceGate` | Maps calibrated confidence → `act` / `confirm` / `escalate`. |
| `ToolRegistry` | Executes tools; resolves args from state/decisions (never model text); enforces idempotency. |
| `ContextProvider` | Pluggable datasource adapters that inject external data into state. |
| `LongTermMemory` | Durable store the harness writes/reads across runs. |
| `Telemetry` + `calibration` | Logs every decision (full distribution) + outcome; supports offline calibration. |
| `Runner` | Owns the loop, budgets, and termination guarantees. |

---

## 3. How each capability is achieved

### 3.1 Iteration

`Runner.run(task)` loops for up to `max_turns`:

1. `questions = policy.questions_for(state)` (two-pass, see §5).
2. `state_text = state_builder.assemble(state)`.
3. `evaluation = client.evaluate(state_text, questions)` (one parallel pass).
4. Append decisions to `state.decisions` (so the model "remembers" past judgments).
5. Gate on confidence; resolve to `Action`s; execute tools; append results as events.
6. Repeat until a **terminal** action, escalation, budget, or no-progress detection.

Because the model has no hidden state, **all continuity comes from the harness** writing
prior decisions and tool results back into the state each turn.

### 3.2 Tool calling (decision-driven, not generative)

Tools are exposed as a **finite, pre-enumerated decision space**, not free-form function
calling:

- The model selects a `next_action` from a `Choice` whose options are tool names / intents.
- **Arguments come from the state or from templates**, e.g. `{{fields.account_id}}`,
  `{{decision.category}}`, `{{task}}` — never from model-generated text.
- Each `Action` binds `(tool, args, terminal)`.

Three patterns (see §6 for when to use each):

- **A. Enumerated routing (pure System One)** — every possible action is an option; args
  are pre-bound or pulled from state. For closed, well-understood workflows.
- **B. Decide-then-fill (System One + LLM)** — System One picks *which* tool and gates
  confidence; a separate LLM generates the open-ended arguments. The LLM is an untrusted
  actor; System One validates its output before execution.
- **C. Retrieval decisions (System One + search)** — System One chooses *what* to fetch
  (among candidate queries/strategies); the harness executes the fetch.

### 3.3 External datasources

Two mechanisms:

- **ContextProviders (passive)** — run *before each evaluate* and inject their output into
  the state (e.g. a knowledge base, current time, account record). They re-run every turn,
  so state is always fresh.
- **Tools (active)** — fetch *and* mutate (e.g. look up an account, issue a refund, POST to
  an API). Writes are the harness's job; the model only decides *whether/when*.

The model never holds credentials and never touches I/O directly — it can only trigger
pre-declared providers/tools.

---

## 4. Confidence-gated control flow

`ConfidenceGate(auto=0.8, escalate=0.5)` maps confidence to:

- `act` — confidence ≥ `auto`: proceed automatically.
- `confirm` — in between: flag for confirmation (interactive `on_confirm` hook, or
  auto-approve in batch mode).
- `escalate` — below `escalate`: route to a human / stop.

By default it uses the **minimum confidence across all questions**, but you should
usually restrict it to the **control-driving questions** (e.g. the routing
`next_action` choice) via `ConfidenceGate(questions=["next_action"])`. Speculative or
informational questions (a root-cause `hypothesis`) are *honestly* low-confidence early
on and must not force an escalation — this was observed directly against the live model,
where `hypothesis` sat at ~0.4 while `next_action` was 0.99.

Escalation and confirmation are **first-class actions** (they appear as options in the
`next_action` choice), not exception paths.

> **Important:** the gate is only as good as the model's calibration. TypeSafe explicitly
> states thresholds are use-case-specific and must be tuned on *your* labeled data.
> Shipping the gate without an outcome-labeling loop (§8) is the single biggest failure
> mode.

---

## 5. Question selection (two-pass)

All questions in one call are answered in parallel against the same state and **cannot
reference each other**. So "route, then ask route-specific questions" is inherently
multi-pass. Per turn, use:

1. **Pass 1 — guards + routing**: `Noul`s (`is_done`, `is_blocked`, `needs_human`)
   plus a routing `Choice` (`next_intent`). Nouls are the lightest-weight primitive;
   use them liberally as guards. (Pricing is input-token-based with free output, so
   per-primitive cost differences are negligible — the saving is clarity, not tokens.)
2. **Pass 2 — route-specific questions**: only the questions relevant to the chosen route.

`Policy.select_questions(state)` lets the harness decide the question set per turn (e.g.
only ask `refundable` once account context is present).

---

## 6. Action-space design (keep routing Choices small)

`Choice` is capped at 255 options; above that, TypeSafe prescribes a two-stage
score-then-choose pattern. As a heuristic, keep routing Choices small anyway: large flat
option sets spread the probability mass thin and are harder to calibrate. We have not
measured the degradation curve ourselves — validate on your labeled data (see §8).
Do **not** expose one flat `next_action` over every tool:

- **Routing Choice** (intents/tool groups, <20) → **per-route argument decisions** (fixed
  schema per tool). Bounds the space to `|routes| + max|args-per-route|`, not the product.
- **Composite scoring** for large option sets: `Score` every candidate independently in one
  parallel pass, then `Choice` over the top-k survivors (TypeSafe's documented two-stage
  pattern).

---

## 7. State budget & memory (the real constraint)

The request budget is ~32k tokens (~150k English chars). Turn history + provider docs +
tool results will exhaust it in a handful of turns. This — not model intelligence — is the
binding constraint. Mitigations in this harness:

- `StateBuilder.max_chars` hard budget with truncation (oldest/largest first).
- **Structured fields kept separate** (`state.fields`) — injected compactly, not stringified
  into prose. (The real API accepts a structured `state` object; the text renderer here is a
  portable simplification.)
- History/event windows (`max_prior_decisions`, `max_history_events`).
- A **relevance filter** can be a System One question itself: the model `Score`s which
  history items are still relevant — the model does its own memory management.

---

## 8. Telemetry & calibration (first-class)

Every turn is logged with the **full probability distribution** (not just argmax), the
chosen actions, the gate, and — crucially — an **outcome label** attached later:

```
telemetry.record(turn, evaluation, gate, actions)
telemetry.attach_outcome(turn, "success" | "failure")   # ground truth, added offline
```

`harness.calibration.expected_calibration_error(records)` then builds reliability
diagnostics so thresholds can be set from data, not vibes. Without outcome labels, the
confidence channel is worthless.

---

## 9. Termination & safety

The model is low-variance across identical calls (TypeSafe: "similar answers for similar
inputs"), so it cannot sample its way out of a stuck loop. The harness bounds the loop
mechanically, and policy authors must follow one rule. Budgets and detection are enforced
in `Runner`; the option-set rule is on you:

- **Monotonic budgets** (enforced): `max_turns`, `max_tool_calls`.
- **No-progress detection** (enforced): if the (decisions + events) fingerprint is unchanged for N
  turns, escalate.
- **Idempotent tool execution** (enforced): a `(tool, args)` pair is never re-run with the same args
  unless explicitly allowed — prevents re-executing side effects in a stuck loop.
- **`done` / `escalate` in every routing Choice** (policy-author rule, not enforced in code):
  there must always be a way out.

Low-confidence retry is allowed **only with augmented state** (more context/history).
Re-asking the identical call is low-value for a low-variance model.

---

## 10. Hybrid LLM bridge (optional, strictly one-directional)

When open-ended text is needed (drafting a reply, generating tool args), delegate to an
LLM — but keep the flow one-directional:

```
System One decides (what + confidence)  →  LLM produces text  →  System One validates (Noul/Score)  →  execute/send
```

The LLM never makes the control-flow decision. The moment it does, the harness collapses
into "just an LLM agent" and the System One model becomes redundant.

---

## 11. API / schema-drift notes

- Endpoint: `POST https://api.typesafe.ai/v1/systemone`, body `{model, state, questions}`.
  Questions are keyed by name with `type` ∈ `{noul, choice, score}` (note: `noul`, not
  `boolean`).
- **Pin the model version** (`jev-1.x.y`) rather than relying on the `jev-latest` alias.
- **Version your question schemas** (`Policy.version`). The `instructions`/`criteria` text is
  the entire decision definition — treat it as a prompt to be versioned and evaluated;
  a poorly phrased criteria list is a silent accuracy loss.
- Parse and log the full distribution + confidence, never just the top label.

---

## 12. Extension points

- `SystemOneClient` — swap TypeSafe for a deterministic mock, or adapt a local model.
  (`DavidHatley/system-one-mini` exists on Hugging Face but is a fixed-head research
  prototype — five preset decisions over short inputs — not a drop-in replacement for
  arbitrary questions; it would need an adapter, not a swap.)
- `ContextProvider` — add `SqlProvider`, `WebSearchProvider`, vector-store RAG, etc.
- `LongTermMemory` — swap the in-memory store for a vector DB.
- `TextGenerator` (LLM bridge) — plug in for pattern B.
- `on_confirm` — plug in human-in-the-loop approval.

See `README.md` for usage; `examples/support_agent.py` is a runnable end-to-end demo.
