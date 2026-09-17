"""Labeled-eval demo: thresholds from data, not vibes (DESIGN.md §8).

Runs the support-ticket mock policy over ``examples/eval_cases.jsonl``,
grades the final decisions per question, attaches ground-truth outcomes to
the telemetry, and grid-searches ConfidenceGate thresholds under an error
budget. Regression gate: exits non-zero if accuracy drops or a run outcome
violates its ``expect_run`` assertion.

Run with:  python3 -m examples.eval_demo

NOTE on the labels in eval_cases.jsonl: they pin the mock policy's *current*
behavior, quirks included, which is exactly what a regression gate is for.
Examples: any run that touches the account DB ends up classified "billing"
in its final turn (the lookup output contains "refundable", which contains
the "refund" keyword), and the support demo's "escalate" action is
terminal-but-not-escalating, so unknown-account runs "complete". Human
ground truth for a production policy would be labeled independently; here
the gate fires when behavior *changes*, and a human then decides whether
the new behavior (and labels) are correct.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import support_agent
from harness import (
    ConfidenceGate,
    FunctionTool,
    Policy,
    Runner,
    StateBuilder,
    ToolRegistry,
    load_cases,
    run_suite,
    sweep_thresholds,
)

CASES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_cases.jsonl")
ACCURACY_FLOOR = 0.8


def make_runner():
    # Same decision space + mock rules as the support demo, but no KB
    # provider (so refund_request follows the ticket + tool outputs only)
    # and sqlite-backed tools.
    support_agent.init_account_db()
    tools = (
        ToolRegistry()
        .register(FunctionTool("lookup_account", support_agent.lookup_account))
        .register(FunctionTool("refund", support_agent.issue_refund))
    )
    policy = Policy(
        support_agent.questions,
        resolvers=[support_agent.resolve],
        select_questions=support_agent.select_questions,
    )
    return Runner(
        support_agent.MockClient(rules=support_agent.mock_rules()),
        policy, tools, StateBuilder(),
        confidence_gate=ConfidenceGate(auto=0.8, escalate=0.5),
        max_turns=6,
    )


def main(cases_path: str | None = None, error_budget: float = 0.15) -> int:
    cases = load_cases(cases_path or CASES_PATH)
    report = run_suite(make_runner, cases)

    print(f"cases: {len(cases)}  question-accuracy: {report.accuracy:.2f}  "
          f"case-accuracy: {report.case_accuracy:.2f}  ECE: {report.ece:.3f}\n")
    failures = 0
    for i, res in enumerate(report.results, 1):
        marks = {q: ("ok" if ok else "WRONG") for q, ok in res.grades.items()}
        run_flag = ""
        if res.case.expect_run is not None and res.run_outcome != res.case.expect_run:
            run_flag = f"  <-- RUN MISMATCH (expected {res.case.expect_run})"
            failures += 1
        failures += sum(1 for ok in res.grades.values() if not ok)
        print(f"case {i:2d} run={res.run_outcome:<16} outcome={res.case.outcome:<8} "
              f"grades={marks}{run_flag}")

    # A 0.15 error budget (not 0.05): the mock's confidences barely
    # discriminate outcomes, so a strict budget degenerates to "act on
    # nothing". The budget itself is a policy choice the table informs.
    tuning = sweep_thresholds(report.records, error_budget=error_budget)
    print(f"\ntuner (error budget {error_budget}): best auto={tuning.best_auto} "
          f"escalate={tuning.best_escalate} coverage={tuning.coverage} "
          f"error_rate={tuning.error_rate}")
    print(f"{'auto':>6} {'esc':>5} {'cover':>6} {'err':>6} {'esc_rate':>8}  n")
    for row in tuning.table:
        star = " *" if (row["auto"] == tuning.best_auto
                        and row["escalate"] == tuning.best_escalate) else ""
        print(f"{row['auto']:>6} {row['escalate']:>5} {row['coverage']:>6} "
              f"{row['error_rate']:>6} {row['escalate_rate']:>8}  "
              f"{row['n_acted']:>2}/{row['n_total']:<2}{star}")

    gate_ok = report.accuracy >= ACCURACY_FLOOR and failures == 0
    print(f"\n{'PASS' if gate_ok else 'FAIL'}: accuracy {report.accuracy:.2f} "
          f"(floor {ACCURACY_FLOOR}), {failures} mismatches")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
