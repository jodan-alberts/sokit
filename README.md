# SOKIT (System One Knowledge Instructions Tools)

A decision-driven agent harness for **System One models** (TypeSafe's Jev-class models).
System One models make fast, calibrated, typed decisions but cannot generate text, call
tools, or fetch data on their own. This harness is the *body* that adds those abilities:
**iterate**, **call tools**, and **access external datasources** — while the model stays a
calibrated policy.

Read **[DESIGN.md](DESIGN.md)** for the full design rationale.

## Layout

```
harness/
  client.py      # SystemOneClient (TypeSafe / mock) — the model adapter
  decisions.py   # Question, Decision, Evaluation (Choice / Score / Noul)
  state.py       # State (working memory) + Event
  context.py     # StateBuilder + ContextProvider protocol + Document
  providers.py   # Files/HTTP/clock/memory/SQL providers (web-search stub)
  policy.py      # Policy, Action (incl. generate-then-validate), route helpers
  tools.py       # ToolRegistry (timeout/retry/error capture, idempotency), FunctionTool
  confidence.py  # ConfidenceGate (act/confirm/escalate)
  memory.py      # LongTermMemory protocol + InMemoryStore + JsonlStore
  telemetry.py   # TurnRecord / Telemetry (decision + outcome logging)
  calibration.py # expected calibration error over labeled runs
  eval.py        # labeled-eval loop: grade_case, run_suite, sweep_thresholds
  generate.py    # TextGenerator bridge (Mock + OpenAI-standard + Anthropic)
  runner.py      # the loop: budgets, no-progress detection, escalation
examples/
  support_agent.py   # end-to-end demo (SQLite-backed account lookup)
  incident_agent.py  # flat-vs-harness incident demo
  draft_reply.py     # generate-then-validate reply drafting (mock path)
  eval_demo.py + eval_cases.jsonl  # labeled-eval regression gate
  cli.py             # interactive CLI (list/run/repl) + tui.py ANSI styling
tests/
  test_runner.py test_tools.py test_datasources.py test_eval.py test_generate.py test_cli.py
```

## Quick start

No install needed for the demo/tests (stdlib only):

```bash
# flat-vs-harness incident demo (real API if TYPESAFE_API_KEY is set)
python3 -m examples.incident_agent
python3 -m examples.incident_agent --mock   # deterministic offline demo

# support-ticket demo (MockClient, SQLite-backed accounts)
python3 -m examples.support_agent

# generate-then-validate drafting demo (mock generator + scripted validator)
python3 -m examples.draft_reply

# labeled-eval regression gate (accuracy/ECE + threshold tuner table)
python3 -m examples.eval_demo

# interactive CLI: list agents, one-shot runs, or a REPL shell
python3 -m examples.cli list
python3 -m examples.cli run support "I want a refund" --fields '{"account_id": "acct_123"}'
python3 -m examples.cli repl draft

# run the tests
python3 -m unittest discover -s tests -v
```

Or install as a package:

```bash
pip install -e .            # base
pip install -e ".[typesafe]" # + requests for the real TypeSafe API
```

## Using the real TypeSafe API

```python
from harness import TypeSafeClient, Runner, Policy, ToolRegistry, StateBuilder, ConfidenceGate

client = TypeSafeClient()  # model="jev-latest"; pin model=... once versions exist
runner = Runner(
    client=client,
    policy=Policy(questions=... , resolvers=[...]),
    tools=ToolRegistry().register(...),
    state_builder=StateBuilder(providers=[...]),
    confidence_gate=ConfidenceGate(auto=0.8, escalate=0.5),
)
result = runner.run(task="...", fields={"account_id": "acct_123"})
```

Set `TYPESAFE_API_KEY` in the environment or in a gitignored `.env` file (loaded
automatically). The API is early-access; adjust `TypeSafeClient._parse` if the
schema drifts.

### Confidence gating

`ConfidenceGate(auto=0.8, escalate=0.5)` maps confidence to `act`/`confirm`/
`escalate`. Gate on the **control-driving questions**, not every question:

```python
ConfidenceGate(auto=0.8, escalate=0.5, questions=["next_action"])
```

Speculative/informational questions (e.g. a root-cause `hypothesis`) are often
*honestly* low-confidence early on and should not force an escalation.

### Tool hardening

Tools are untrusted: exceptions degrade to `ERROR` results and hangs to
`timeout` results instead of killing the loop, with retries + exponential
backoff on infra failures only (`ToolRegistry(default_timeout=30.0,
default_retries=0)`, per-tool overrides on `FunctionTool`). Only `ok=True`
results enter the idempotency cache, so failed calls stay re-runnable; `N`
consecutive failures (`Runner(..., max_tool_errors=3)`) escalate.

### Datasources & memory

`SqlProvider` is a read-only SQLite datasource (stdlib `sqlite3`, `mode=ro` +
SELECT/WITH-only, separately-bound params); `JsonlStore` is crash-tolerant
file-backed long-term memory (single-writer). The support demo seeds its
accounts into a temp SQLite file at startup.

### Labeled eval

`harness/eval.py` grades labeled JSONL cases, attaches outcomes to telemetry,
and tunes gate thresholds under an error budget — see `DESIGN.md` §8 and
`python3 -m examples.eval_demo`.

### LLM bridge (pattern B)

System One decides, an LLM drafts, System One validates before execution:

```python
Action("draft", tool="send_reply", args={"account_id": "{{fields.account_id}}"},
       generate={"slot": "message",
                 "prompt": "Draft a refund reply for {{fields.account_id}}: {{task}}",
                 "validator": "Is this draft safe, correct, and on-policy?"})
runner = Runner(..., generator=MockGenerator(template="..."))  # offline
```

For a real LLM, pick a generator (keys from env — never flags, so they stay
out of shell history):

```python
from harness import AnthropicGenerator, HttpGenerator

Runner(..., generator=HttpGenerator())  # OpenAI (OPENAI_API_KEY)
Runner(..., generator=HttpGenerator(    # OpenRouter (OpenAI-standard)
    base_url="https://openrouter.ai/api/v1", model="openai/gpt-4o-mini",
    extra_headers={"HTTP-Referer": "https://myapp.test", "X-Title": "sokit"}))
Runner(..., generator=HttpGenerator(    # local OpenAI-standard server
    base_url="http://localhost:11434/v1", model="llama3", api_key="ollama"))
Runner(..., generator=AnthropicGenerator(model="claude-..."))  # ANTHROPIC_API_KEY
```

Or from the CLI (the `draft` agent showcasing the bridge end to end):

```bash
python3 -m examples.cli run draft "I want a refund" \
  --fields '{"account_id": "acct_123"}' --generator openai
python3 -m examples.cli run draft "I want a refund" \
  --fields '{"account_id": "acct_123"}' --generator anthropic \
  --generator-model claude-sonnet-4-5
python3 -m examples.cli run draft "I want a refund" \
  --fields '{"account_id": "acct_123"}' --generator openai \
  --base-url https://openrouter.ai/api/v1 --generator-model openai/gpt-4o-mini
```

See `python3 -m examples.draft_reply` and `DESIGN.md` §10.

### CLI

`examples/cli.py` (plus `examples/tui.py` ANSI styling, stdlib only) is the
interactive entry point for demoing, playing with, and testing the agents:

```bash
python3 -m examples.cli list            # support · incident · draft
python3 -m examples.cli run incident "pager" --mock
python3 -m examples.cli repl                       # defaults to incident
python3 -m examples.cli repl support
```

**Client selection.** `--mock` forces the deterministic mock, `--real`
forces the live TypeSafe API (errors if `TYPESAFE_API_KEY` is missing).
Default: live when a key is set, mock otherwise (with a stderr notice).
Flags work globally or per-subcommand.

**`run` flags.** Positional `agent` + `task`; `--fields '{"k": "v"}'`
(JSON object), `--auto/--escalate` gate overrides (the factory gate's
question scoping is preserved), `--max-turns`, `--quiet` (banner only),
`--verbose` (question definitions + state preview on top of the default
trace), `--yes` (auto-approve confirm gates), `--telemetry-out run.jsonl`,
`--transcript-out session.log`, and `--generator mock|openai|anthropic`
with `--generator-model`/`--base-url` for the `draft` agent. Exit codes:
`0` completed, `1` any other outcome, `2` usage error (unknown agent, bad
`--fields`, missing LLM key).

**Understanding the model.** Every turn shows the decision *plus* what the
model considered and why the gate fired: the top-3 probability mass per
question (so the runner-up is visible), the top-2 margin (decisiveness),
and a gate line naming the min-confidence driver, thresholds, and scope
(`scoped to [next_action]` vs all questions). Runs open with the model
identity (`mock (deterministic, offline)` vs live TypeSafe model +
endpoint) and close with a summary (policy version, turns, tool
calls/errors, final values with runners-up, last event). `--verbose`
adds each question's type + instructions + options and a truncated
preview of the assembled state the model actually read. `show
<run.jsonl>` re-renders the same distributions from saved telemetry.

**`repl` commands.** Type a task to run it (fresh runner per task, so no
idempotency/memory bleed across runs). Every agent prints a short guide on
start/switch (`:guide` reprints it) with example tasks worth trying. Session
commands:

```
:fields {...}  set persistent fields   :agent <name>  switch agent
:gates <auto> <esc>  retune live       :trace on|off  :guide
:inspect [agent]  decision space       :questions  list questions
:verbose on|off                       :retry  rerun last task
:task  multi-line task (end with .)   :help  :quit
```

Tab-completion for commands/agents and persistent history
(`~/.sokit_history`, override via `SOKIT_HISTORY`) where readline exists.

CONFIRM gates prompt `y/n` on stdin (the `on_confirm` hook); `--yes`
batch-approves. Colors auto-disable when piped or under `NO_COLOR`.

**More commands.** `doctor` checks the environment (Python version, API
keys, sqlite3, endpoint reachability, readline); `eval` runs the
labeled-eval regression gate (`--cases`, `--error-budget`); `inspect
<agent>` shows the full decision space (questions + options, tools, gate,
policy version); `show <run.jsonl>` re-renders saved telemetry with
distributions.

```bash
python3 -m examples.cli doctor
python3 -m examples.cli eval
python3 -m examples.cli inspect support --mock
python3 -m examples.cli show run.jsonl
```

## Minimal example

```python
from harness import (
    Action, MockClient, Policy, Runner, StateBuilder, ToolRegistry,
    FunctionTool, ToolResult, choice, noul,
)

questions = {
    "spam": noul("Is this email spam?"),
    "next": choice("What next?", ["file", "delete", "done"]),
}

def resolve(evaluation, state):
    nxt = evaluation["next"].value
    if nxt == "delete":
        return [Action("delete", tool="delete_email", args={"id": "{{fields.id}}"})]
    return [Action("done", terminal=True)]

def delete_email(args, context):
    return ToolResult("delete_email", True, f"deleted {args['id']}")

client = MockClient(rules={
    "spam": {"yes": ["buy now", "free money", "winner", "act now"], "no": []},
    "next": {"delete": ["spam"], "done": ["meeting"]},
})
tools = ToolRegistry().register(FunctionTool("delete_email", delete_email))
runner = Runner(client, Policy(questions, resolvers=[resolve]), tools,
                StateBuilder(), max_turns=4)

result = runner.run(task="Email: 'BUY NOW free money'", fields={"id": 42})
print(result.outcome)
for r in result.telemetry.records:
    print(r.turn, r.gate, {q: d["value"] for q, d in r.decisions.items()})
```
