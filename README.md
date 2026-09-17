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
  providers.py   # Files/HTTP/clock/memory providers (SQL & web-search stubs)
  policy.py      # Policy, Action, route helpers
  tools.py       # ToolRegistry, FunctionTool, argument templating
  confidence.py  # ConfidenceGate (act/confirm/escalate)
  memory.py      # LongTermMemory protocol + in-memory store
  telemetry.py   # TurnRecord / Telemetry (decision + outcome logging)
  calibration.py # expected calibration error over labeled runs
  runner.py      # the loop: budgets, no-progress detection, escalation
examples/
  support_agent.py   # end-to-end demo
tests/
  test_runner.py
```

## Quick start

No install needed for the demo/tests (stdlib only):

```bash
# flat-vs-harness incident demo (real API if TYPESAFE_API_KEY is set)
python3 -m examples.incident_agent
python3 -m examples.incident_agent --mock   # deterministic offline demo

# support-ticket demo (MockClient)
python3 -m examples.support_agent

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

client = TypeSafeClient()  # default model is jev-latest; pin a released version once available
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
