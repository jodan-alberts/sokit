"""Labeled evaluation loop (DESIGN.md §8).

Thresholds from data, not vibes: run the harness over hand-labeled cases,
grade the final decisions per question, attach ground-truth outcomes to the
telemetry records (which feeds ``expected_calibration_error``), and
grid-search ``ConfidenceGate`` thresholds under an error budget.

Case format (JSONL, one per line)::

    {"task": "...", "fields": {"account_id": "acct_123"},
     "expected": {"category": "billing", "refund_request": true},
     "outcome": "success"}

Score questions are graded with a tolerance; Choice/Noul by exact match.
Only questions listed in ``expected`` are graded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .calibration import expected_calibration_error
from .decisions import Evaluation, QuestionType
from .telemetry import TurnRecord

if TYPE_CHECKING:  # avoid a runtime import cycle (runner imports telemetry too)
    from .runner import Runner


@dataclass
class EvalCase:
    task: str
    fields: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"          # ground-truth run label attached to telemetry
    tolerance: float = 1e-6           # score-question grading tolerance
    expect_run: str | None = None     # optional regression assertion on Runner outcome


@dataclass
class CaseResult:
    case: EvalCase
    evaluation: Evaluation | None
    run_outcome: str                  # "completed" | "escalated" | ...
    grades: dict[str, bool]
    correct: bool                     # every expected question graded correct
    records: list[TurnRecord]


@dataclass
class SuiteReport:
    results: list[CaseResult]
    records: list[TurnRecord]         # flat per-turn records, outcomes attached
    accuracy: float                   # question-level fraction correct
    case_accuracy: float              # fraction of cases fully correct
    ece: float


@dataclass
class ThresholdTuning:
    best_auto: float
    best_escalate: float
    coverage: float
    error_rate: float
    table: list[dict[str, Any]]


def _noul_expected(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("yes", "true", "1", "y")
    return bool(value)


def grade_question(decision: Any, expected: Any, tolerance: float) -> bool:
    """Grade one decision against its expected value.

    Score uses ``tolerance``; Choice/Noul use exact match (Noul coerces
    yes/no-style strings to bool). A missing decision grades incorrect.
    """
    if decision is None:
        return False
    if decision.type is QuestionType.SCORE:
        try:
            return abs(float(decision.value) - float(expected)) <= tolerance
        except (TypeError, ValueError):
            return False
    if decision.type is QuestionType.NOUL:
        return bool(decision.value) is _noul_expected(expected)
    return decision.value == expected


def grade_case(evaluation: Evaluation | None, case: EvalCase) -> dict[str, bool]:
    """Per-question correct/incorrect for the final evaluation of a run."""
    grades: dict[str, bool] = {}
    for question, expected in case.expected.items():
        decision = evaluation.get(question) if evaluation is not None else None
        grades[question] = grade_question(decision, expected, case.tolerance)
    return grades


def run_suite(runner_factory: Callable[[], "Runner"],
              cases: list[EvalCase]) -> SuiteReport:
    """Run each case on a fresh runner (factory called per case, so
    idempotency caches and tool state never leak across cases).

    Every turn record gets the case's ground-truth ``outcome`` attached,
    making the flat record list directly usable by
    ``expected_calibration_error`` and ``sweep_thresholds``.
    """
    results: list[CaseResult] = []
    records: list[TurnRecord] = []
    for case in cases:
        runner = runner_factory()
        result = runner.run(case.task, dict(case.fields))
        for record in result.telemetry.records:
            record.outcome = case.outcome
        grades = grade_case(result.evaluation, case)
        results.append(CaseResult(
            case=case,
            evaluation=result.evaluation,
            run_outcome=result.outcome,
            grades=grades,
            correct=all(grades.values()) if grades else True,
            records=list(result.telemetry.records),
        ))
        records.extend(result.telemetry.records)

    total_q = sum(len(r.grades) for r in results)
    correct_q = sum(1 for r in results for ok in r.grades.values() if ok)
    return SuiteReport(
        results=results,
        records=records,
        accuracy=(correct_q / total_q) if total_q else 1.0,
        case_accuracy=(sum(1 for r in results if r.correct) / len(results)) if results else 1.0,
        ece=expected_calibration_error(records),
    )


def load_cases(path: str) -> list[EvalCase]:
    """Read a JSONL file of cases (one JSON object per line)."""
    cases: list[EvalCase] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(EvalCase(
                task=obj.get("task", ""),
                fields=obj.get("fields", {}),
                expected=obj.get("expected", {}),
                outcome=obj.get("outcome", "success"),
                tolerance=obj.get("tolerance", 1e-6),
                expect_run=obj.get("expect_run"),
            ))
    return cases


def _record_confidence(record: TurnRecord) -> float:
    if not record.decisions:
        return 0.0
    return min(d["confidence"] for d in record.decisions.values())


def sweep_thresholds(
    records: list[TurnRecord],
    error_budget: float = 0.05,
    auto_grid: list[float] | None = None,
    escalate_grid: list[float] | None = None,
) -> ThresholdTuning:
    """Grid-search ``ConfidenceGate(auto, escalate)`` thresholds.

    Simulation: a turn with min-confidence >= ``auto`` would have acted on
    its own (``coverage`` = fraction of such turns); ``error_rate`` is the
    failure rate among those acted turns (outcome != "success").
    ``escalate_rate`` (fraction below ``escalate``) is reported per cell as
    the confirm-band tradeoff — it cannot be tuned from act-outcomes alone,
    and the table makes that visible.

    Returns the highest-coverage pair with ``error_rate <= error_budget``
    (ties: stricter ``auto``, then lower ``escalate``). Cells that would act
    on nothing (``n_acted == 0``) are reported but never selected — a
    threshold that acts on zero turns is not a usable recommendation. If no
    acting cell is under budget, returns the lowest-error acting pair
    instead (and if there are no usable records at all, a zero default).
    """
    usable = [r for r in records if r.outcome is not None and r.decisions]
    auto_grid = auto_grid or [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    escalate_grid = escalate_grid or [0.3, 0.4, 0.5, 0.6, 0.7]

    table: list[dict[str, Any]] = []
    for auto in auto_grid:
        acted = [r for r in usable if _record_confidence(r) >= auto]
        coverage = len(acted) / len(usable) if usable else 0.0
        error_rate = (
            sum(1 for r in acted if r.outcome != "success") / len(acted)
            if acted else 0.0
        )
        for esc in escalate_grid:
            if esc > auto:
                continue
            escalate_rate = (
                sum(1 for r in usable if _record_confidence(r) < esc) / len(usable)
                if usable else 0.0
            )
            table.append({
                "auto": auto, "escalate": esc,
                "coverage": round(coverage, 3), "error_rate": round(error_rate, 3),
                "escalate_rate": round(escalate_rate, 3),
                "n_acted": len(acted), "n_total": len(usable),
            })

    under = [row for row in table if row["n_acted"] > 0 and row["error_rate"] <= error_budget]
    if under:
        best = min(under, key=lambda r: (-r["coverage"], -r["auto"], r["escalate"]))
    else:
        acting = [row for row in table if row["n_acted"] > 0]
        pool = acting or [{"auto": 0.8, "escalate": 0.5, "coverage": 0.0,
                           "error_rate": 0.0, "escalate_rate": 0.0,
                           "n_acted": 0, "n_total": len(usable)}]
        best = min(pool, key=lambda r: (r["error_rate"], -r["coverage"]))
    return ThresholdTuning(
        best_auto=best["auto"], best_escalate=best["escalate"],
        coverage=best["coverage"], error_rate=best["error_rate"], table=table,
    )
