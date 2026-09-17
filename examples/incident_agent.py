"""Incident remediation — a flow that only works with the harness, not a flat model.

The scenario: a 2am pager alert for the checkout API. A "flat" System One call
can classify the alert but cannot act on it — it has no way to check health, see
deploys, roll back, or re-evaluate. The harness gives the same model a body:
tools + datasources + a loop, so it investigates, correlates, acts, verifies,
and either resolves or escalates with evidence.

Run:
    python3 -m examples.incident_agent            # real API (uses .env key)
    python3 -m examples.incident_agent --mock     # deterministic offline demo
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import (
    Action,
    ClockProvider,
    ConfidenceGate,
    Decision,
    Document,
    FunctionTool,
    InMemoryStore,
    MockClient,
    Policy,
    Runner,
    StateBuilder,
    ToolRegistry,
    ToolResult,
    TypeSafeClient,
    choice,
    noul,
)
from harness.client import _read_env_file

# --- the world (mutable environment the tools act on) -------------------------
TASK = ("PagerDuty: checkout API returning HTTP 500 for 12 minutes. "
        "Error rate 40%. Started 02:05.")

DEPLOY_LOG = [{"id": "4821", "time": "02:10", "component": "checkout", "author": "alice"}]

RUNBOOK = (
    "RUNBOOK for checkout 5xx incidents:\n"
    "1. Check current service health.\n"
    "2. List recent deploys to the checkout service.\n"
    "3. If a deploy landed just before the incident and the service is down, roll it back.\n"
    "4. Re-check health after the rollback.\n"
    "5. If the service recovered, resolve. If still down, page on-call with the evidence."
)

ENV = {"health": "down", "rolled_back": False}
FIX_WORKS = True


# --- tools (the harness's hands) ----------------------------------------------
def check_health(args, context):
    return ToolResult("check_health", True, f"SERVICE HEALTH: {ENV['health']}")


def list_deploys(args, context):
    lines = "; ".join(f"deploy {d['id']} at {d['time']} on {d['component']} by {d['author']}"
                      for d in DEPLOY_LOG)
    return ToolResult("list_deploys", True, f"RECENT DEPLOYS: {lines}")


def rollback(args, context):
    deploy_id = args.get("deploy_id", "4821")
    if deploy_id != "4821":
        return ToolResult("rollback", False, f"unknown deploy {deploy_id}")
    ENV["rolled_back"] = True
    if FIX_WORKS:
        ENV["health"] = "up"
    return ToolResult("rollback", True,
                      f"ROLLBACK: rolled back deploy {deploy_id}. SERVICE HEALTH NOW: {ENV['health']}")


def page_oncall(args, context):
    message = args.get("message", "incident")
    return ToolResult("page_oncall", True, "PAGED ON-CALL: engineer notified",
                      data={"escalated": message})


# --- the decision space (what the model can "think") ---------------------------
QUESTIONS = {
    "is_severe": noul("Is this a severe production incident needing immediate action?"),
    "hypothesis": choice("What is the most likely root cause of the incident?", {
        "recent_deploy": "A recent deployment broke the service",
        "infra": "Infrastructure or hardware problem",
        "dependency": "A third-party dependency failed",
        "unknown": "Not enough evidence yet",
    }),
    "next_action": choice("What is the single next action to take?", {
        "check_health": "Query the current health of the checkout service",
        "list_deploys": "List recent deployments to the checkout service",
        "rollback": "Roll back the most recent checkout deploy (only when a recent deploy "
                    "is the likely cause and the service is down)",
        "page_oncall": "Escalate to a human engineer with the evidence gathered so far",
        "resolve": "Mark the incident resolved (only when the service is confirmed healthy)",
    }),
}


def resolve(evaluation, state):
    nxt = evaluation["next_action"].value
    if nxt == "check_health":
        return [Action("check", tool="check_health")]
    if nxt == "list_deploys":
        return [Action("list", tool="list_deploys")]
    if nxt == "rollback":
        return [Action("rollback", tool="rollback", args={"deploy_id": "4821"})]
    if nxt == "page_oncall":
        return [Action("page", tool="page_oncall", args={"message": "{{task}}"}),
                Action("escalate", terminal=True, escalate=True)]
    if nxt == "resolve":
        return [Action("resolve", terminal=True)]
    return [Action("escalate", terminal=True, escalate=True)]


def runbook_provider(state):
    return [Document("runbook", RUNBOOK)]


# --- mock rules (deterministic offline narrative) ------------------------------
def _decide(name, question, value, conf):
    options = question.options
    probs = {o: (0.9 if o == value else 0.1 / (len(options) - 1)) for o in options}
    return Decision(name, question.type, value, probabilities=probs, confidence=conf)


def _mock_hypothesis(state_text, question):
    if "deploy 4821" in state_text.lower():
        return _decide("hypothesis", question, "recent_deploy", 0.92)
    return _decide("hypothesis", question, "unknown", 0.55)


def _mock_next_action(state_text, question):
    s = state_text.lower()
    if "service health now: up" in s:
        return _decide("next_action", question, "resolve", 0.97)
    if "service health now: down" in s:
        return _decide("next_action", question, "page_oncall", 0.85)
    if "service health: down" in s and "deploy 4821" in s:
        return _decide("next_action", question, "rollback", 0.90)
    if "service health: down" in s and "deploy 4821" not in s:
        return _decide("next_action", question, "list_deploys", 0.90)
    if "service health:" not in s:
        return _decide("next_action", question, "check_health", 0.95)
    return _decide("next_action", question, "page_oncall", 0.70)


MOCK_RULES = {
    "is_severe": {"yes": ["500", "down", "error", "incident"], "no": []},
    "hypothesis": _mock_hypothesis,
    "next_action": _mock_next_action,
}


# --- the flow ------------------------------------------------------------------
def make_runner(client):
    tools = (
        ToolRegistry()
        .register(FunctionTool("check_health", check_health))
        .register(FunctionTool("list_deploys", list_deploys))
        .register(FunctionTool("rollback", rollback))
        .register(FunctionTool("page_oncall", page_oncall))
    )
    memory = InMemoryStore()
    state_builder = StateBuilder(providers=[runbook_provider, ClockProvider()])
    policy = Policy(QUESTIONS, resolvers=[resolve])
    runner = Runner(
        client, policy, tools, state_builder,
        confidence_gate=ConfidenceGate(auto=0.8, escalate=0.5, questions=["next_action"]),
        longterm_memory=memory,
        max_turns=8,
        no_progress_limit=3,
    )
    return runner, memory


def run_flat(client):
    """A single evaluate() call — the whole story for a flat System One model."""
    state_builder = StateBuilder(providers=[runbook_provider, ClockProvider()])
    from harness.state import State
    text = state_builder.assemble(State(task=TASK))
    evaluation = client.evaluate(text, QUESTIONS)
    for name, d in evaluation.decisions.items():
        print(f"    {name}: {d.value!r}  (conf {d.confidence:.2f})")
    print()
    print("    The flat model emits its judgment and stops. There is no executor, no")
    print("    follow-up call, no way to act or learn. A human must run the tool,")
    print("    paste the result back, and re-call the model — by hand, every turn.")
    print("    The harness automates exactly that loop.\n")


def print_trace(result):
    for record in result.telemetry.records:
        d = record.decisions
        sev = d.get("is_severe", {}).get("value")
        hyp = d.get("hypothesis", {}).get("value")
        nxt = d.get("next_action", {})
        print(f"    turn {record.turn}  gate={record.gate:<7} severe={sev} "
              f"hypothesis={hyp:<13} next={nxt.get('value')}  (next conf {nxt.get('confidence', 0.0):.2f})")
        for e in result.state.events:
            if e.turn == record.turn:
                print(f"             -> tool {e.source}: {e.content}")
    print(f"    OUTCOME: {result.outcome}\n")


def run_harness(client, fix_works):
    global FIX_WORKS
    FIX_WORKS = fix_works
    ENV["health"] = "down"
    ENV["rolled_back"] = False
    runner, memory = make_runner(client)
    result = runner.run(task=TASK)
    print_trace(result)
    if memory.items:
        print(f"    long-term memory (evidence persisted): {[i['content'] for i in memory.items]}\n")
    return result


def main():
    use_mock = "--mock" in sys.argv
    _read_env_file()
    has_key = bool(os.environ.get("TYPESAFE_API_KEY"))
    if use_mock or not has_key:
        client = MockClient(rules=MOCK_RULES)
        label = "MOCK (offline)" if use_mock else "MOCK (no TYPESAFE_API_KEY found)"
    else:
        client = TypeSafeClient()
        label = f"REAL TypeSafe ({client.model})"

    print("=" * 72)
    print("INCIDENT: checkout API down at 02:05 — pager fired")
    print(f"client: {label}")
    print("=" * 72)

    print("\n[1] FLAT System One model (one call, no harness)")
    print("-" * 72)
    run_flat(client)

    print("\n[2] HARNESS — rollback fixes the outage")
    print("-" * 72)
    run_harness(client, fix_works=True)

    print("\n[3] HARNESS — rollback does NOT fix it (fallback path)")
    print("-" * 72)
    run_harness(client, fix_works=False)


if __name__ == "__main__":
    main()
