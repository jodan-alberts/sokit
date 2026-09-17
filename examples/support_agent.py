"""End-to-end example: an automated support-ticket agent.

Demonstrates all three capabilities the harness adds around a System One model
(which on its own only makes decisions and cannot act):

  1. iterate     — the Runner loops for several turns
  2. call tools  — lookup_account and refund tools run under model control
  3. datasources — the account DB is fetched by a tool; a knowledge base is
                   injected by a context provider every turn

Run with:  python -m examples.support_agent
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import (
    Action,
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
    choice,
    noul,
)

# --- external datasources -----------------------------------------------------
# Seed data; materialized into a small SQLite file at startup so the account
# lookup exercises the real SqlProvider path (still hermetic: temp file, mock
# client, no network).
ACCOUNTS = {
    "acct_123": {"plan": "pro", "refundable": True},
    "acct_999": {"plan": "basic", "refundable": False},
}
KB = "Knowledge base: refunds are allowed for 'pro' plans within 30 days."

_DB_PATH = os.path.join(tempfile.gettempdir(), "sokit_support_accounts.db")


def init_account_db(path: str = _DB_PATH) -> str:
    """Create (or recreate) the demo account database from ACCOUNTS."""
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY, plan TEXT, refundable INTEGER)")
        conn.executemany(
            "INSERT INTO accounts (id, plan, refundable) VALUES (?, ?, ?)",
            [(aid, a["plan"], int(a["refundable"])) for aid, a in ACCOUNTS.items()],
        )
        conn.commit()
    finally:
        conn.close()
    return path


def get_account(account_id: str) -> dict | None:
    """Read one account from the SQLite file (falls back to ACCOUNTS pre-init)."""
    if os.path.exists(_DB_PATH):
        conn = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT plan, refundable FROM accounts WHERE id = ?", (account_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {"plan": row["plan"], "refundable": bool(row["refundable"])}
    return ACCOUNTS.get(account_id)


def kb_provider(state):
    # passive datasource: injected into state before every evaluate()
    return [Document("kb", KB)]


# --- tools --------------------------------------------------------------------
def lookup_account(args, context):
    account_id = args.get("account_id", "")
    acct = get_account(account_id)
    if acct is None:
        return ToolResult("lookup_account", False, f"account {account_id} not found")
    return ToolResult(
        "lookup_account", True,
        f"account {account_id}: plan={acct['plan']} refundable={acct['refundable']}",
    )


def issue_refund(args, context):
    account_id = args.get("account_id", "")
    acct = get_account(account_id)
    if acct is None:
        return ToolResult("refund", False, f"account {account_id} not found")
    if not acct["refundable"]:
        return ToolResult("refund", False, "account is not refundable")
    return ToolResult("refund", True, "refund issued",
                      data={"account": account_id, "status": "refunded"})


# --- the decision space (what the model can "think") ---------------------------
questions = {
    "category": choice("What is the category of this ticket?",
                       ["billing", "technical", "account", "other"]),
    "refund_request": noul("Does the customer want a refund?"),
    "refundable": noul("Is the account eligible for a refund based on the account context?"),
    "next_action": choice("What should the agent do next?",
                          ["lookup_account", "refund", "escalate", "done"]),
}


def select_questions(state):
    """Two-pass selection: only ask the account-specific question once we have
    account context in state (all questions in one call are parallel/independent,
    so route-then-specialize is inherently multi-pass)."""
    qs = {k: v for k, v in questions.items() if k != "refundable"}
    if any(e.source == "lookup_account" for e in state.events):
        qs["refundable"] = questions["refundable"]
    return qs


def _choice_dec(name, question, value, conf):
    options = question.options
    probs = {o: (0.9 if o == value else 0.1 / (len(options) - 1)) for o in options}
    return Decision(name, question.type, value, probabilities=probs, confidence=conf)


def next_action_rule(state, question):
    """Policy-as-code: choose the next action from the current state text."""
    s = state.lower()
    if "refund issued" in s:
        return _choice_dec("next_action", question, "done", 0.97)
    if "account acct_" not in s:
        return _choice_dec("next_action", question, "lookup_account", 0.95)
    if "refundable=true" in s and "refund" in s:
        return _choice_dec("next_action", question, "refund", 0.92)
    return _choice_dec("next_action", question, "escalate", 0.6)


def resolve(evaluation, state):
    nxt = evaluation["next_action"].value
    if nxt == "lookup_account":
        return [Action("lookup", tool="lookup_account", args={"account_id": "{{fields.account_id}}"})]
    if nxt == "refund":
        return [Action("refund", tool="refund", args={"account_id": "{{fields.account_id}}"})]
    if nxt == "done":
        return [Action("done", terminal=True)]
    return [Action("escalate", terminal=True)]


def mock_rules():
    """The deterministic MockClient rules (shared with examples/eval_demo.py)."""
    return {
        "category": {
            "billing": ["refund", "charge", "invoice", "billing"],
            "account": ["account"],
            "technical": ["error", "crash"],
        },
        "refund_request": {"yes": ["refund", "money back"], "no": []},
        "refundable": {"yes": ["refundable=true"], "no": ["refundable=false"]},
        "next_action": next_action_rule,
    }


def make_runner(client):
    """Build the support-ticket runner (shared with examples/cli.py)."""
    init_account_db()
    tools = (
        ToolRegistry()
        .register(FunctionTool("lookup_account", lookup_account))
        .register(FunctionTool("refund", issue_refund))
    )
    state_builder = StateBuilder(providers=[kb_provider])
    gate = ConfidenceGate(auto=0.8, escalate=0.5)
    memory = InMemoryStore()
    policy = Policy(questions, resolvers=[resolve], select_questions=select_questions)
    runner = Runner(client, policy, tools, state_builder,
                    confidence_gate=gate, longterm_memory=memory, max_turns=6)
    return runner, memory


def main():
    client = MockClient(rules=mock_rules())
    runner, memory = make_runner(client)

    result = runner.run(
        task="Customer says: 'I want a refund for my subscription, please.'",
        fields={"account_id": "acct_123"},
    )

    print(f"outcome: {result.outcome}\n")
    for record in result.telemetry.records:
        print(f"turn {record.turn}  gate={record.gate}  note={record.note}")
        for q, d in record.decisions.items():
            print(f"    {q}: {d['value']}  (conf {d['confidence']:.2f})")
        for a in record.actions:
            suffix = f" tool={a['tool']} args={a['args']}" if a["tool"] else " (terminal)"
            print(f"    -> {a['name']}{suffix}")
        print()

    print("long-term memory:")
    for item in memory.items:
        print("   ", item["content"])


if __name__ == "__main__":
    main()
