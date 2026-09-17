"""Generate-then-validate example (§10, pattern B) on the mock path.

A support ticket needs an open-ended reply — something a System One model
cannot write. The flow stays one-directional:

    System One decides (lookup, then draft_reply)  →  MockGenerator drafts
    →  System One validates (draft_ok Noul)  →  send_reply executes

The generator only fills ``args["message"]``; it never chooses the action,
and the validator verdict never re-enters routing.

Run with:  python3 -m examples.draft_reply
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import support_agent
from harness import (
    Action,
    ConfidenceGate,
    Decision,
    FunctionTool,
    MockClient,
    MockGenerator,
    Policy,
    Runner,
    StateBuilder,
    ToolRegistry,
    ToolResult,
    choice,
)


def send_reply(args, context):
    return ToolResult("send_reply", True,
                      f"message sent to {args.get('account_id', '')}: {args.get('message', '')}",
                      data={"sent": args.get("message", "")})


def _decide(name, question, value, conf):
    options = question.options
    probs = {o: (0.9 if o == value else 0.1 / (len(options) - 1)) for o in options}
    return Decision(name, question.type, value, probabilities=probs, confidence=conf)


def next_action_rule(state_text, question):
    s = state_text.lower()
    if "message sent" in s:
        return _decide("next_action", question, "done", 0.97)
    if "account acct_" not in s:
        return _decide("next_action", question, "lookup_account", 0.95)
    return _decide("next_action", question, "draft_reply", 0.92)


questions = {
    "next_action": choice("What should the agent do next?",
                          ["lookup_account", "draft_reply", "done"]),
}


def resolve(evaluation, state):
    nxt = evaluation["next_action"].value
    if nxt == "lookup_account":
        return [Action("lookup", tool="lookup_account",
                       args={"account_id": "{{fields.account_id}}"})]
    if nxt == "draft_reply":
        return [Action("draft", tool="send_reply",
                       args={"account_id": "{{fields.account_id}}"},
                       generate={
                           "slot": "message",
                           "prompt": "Draft a refund reply for {{fields.account_id}}: {{task}}",
                           "validator": "Is this draft safe, correct, and on-policy?",
                       })]
    return [Action("done", terminal=True)]


def make_runner(client, generator):
    """Build the draft-reply runner (shared with examples/cli.py)."""
    support_agent.init_account_db()
    tools = (
        ToolRegistry()
        .register(FunctionTool("lookup_account", support_agent.lookup_account))
        .register(FunctionTool("send_reply", send_reply))
    )
    return Runner(client, Policy(questions, resolvers=[resolve]), tools,
                  StateBuilder(), confidence_gate=ConfidenceGate(auto=0.8, escalate=0.5),
                  generator=generator, max_turns=6)


def main():
    client = MockClient(rules={
        "next_action": next_action_rule,
        "draft_ok": {"yes": ["kind regards"], "no": []},
    })
    generator = MockGenerator(
        template="Hello! Your refund is on its way. Kind regards, Support")
    runner = make_runner(client, generator)
    result = runner.run(
        task="Customer says: 'I want a refund for my subscription, please.'",
        fields={"account_id": "acct_123"},
    )
    print(f"outcome: {result.outcome}\n")
    for record in result.telemetry.records:
        print(f"turn {record.turn}  gate={record.gate}  note={record.note}")
        for q, d in record.decisions.items():
            print(f"    {q}: {d['value']}  (conf {d['confidence']:.2f})")
    print()
    for e in result.state.events:
        print(f"turn {e.turn}  [{e.kind}:{e.source}] {e.content}")


if __name__ == "__main__":
    main()
