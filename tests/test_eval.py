"""Tests for Phase 3 labeled eval: grading, run_suite, threshold tuner."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import (
    Action,
    Decision,
    EvalCase,
    Evaluation,
    MockClient,
    Policy,
    QuestionType,
    Runner,
    StateBuilder,
    ToolRegistry,
    choice,
    grade_case,
    load_cases,
    noul,
    run_suite,
    score,
    sweep_thresholds,
)
from harness.telemetry import TurnRecord


def _dec(name, qtype, value, conf=0.9):
    return Decision(name, qtype, value, probabilities={}, confidence=conf)


class TestGrading(unittest.TestCase):
    def test_choice_exact_match(self):
        ev = Evaluation({"c": _dec("c", QuestionType.CHOICE, "billing")})
        self.assertEqual(grade_case(ev, EvalCase("t", expected={"c": "billing"})), {"c": True})
        self.assertEqual(grade_case(ev, EvalCase("t", expected={"c": "technical"})), {"c": False})

    def test_noul_bool_and_string(self):
        ev = Evaluation({"n": _dec("n", QuestionType.NOUL, True)})
        self.assertTrue(grade_case(ev, EvalCase("t", expected={"n": True}))["n"])
        self.assertTrue(grade_case(ev, EvalCase("t", expected={"n": "yes"}))["n"])
        self.assertFalse(grade_case(ev, EvalCase("t", expected={"n": False}))["n"])

    def test_score_tolerance(self):
        ev = Evaluation({"s": _dec("s", QuestionType.SCORE, 0.75)})
        self.assertTrue(
            grade_case(ev, EvalCase("t", expected={"s": 0.8}, tolerance=0.1))["s"])
        self.assertFalse(
            grade_case(ev, EvalCase("t", expected={"s": 0.8}, tolerance=0.01))["s"])

    def test_missing_decision_is_incorrect(self):
        self.assertEqual(grade_case(Evaluation({}), EvalCase("t", expected={"c": "x"})),
                         {"c": False})
        self.assertEqual(grade_case(None, EvalCase("t", expected={"c": "x"})), {"c": False})

    def test_only_expected_questions_graded(self):
        ev = Evaluation({"a": _dec("a", QuestionType.CHOICE, "x"),
                         "b": _dec("b", QuestionType.CHOICE, "y")})
        self.assertEqual(set(grade_case(ev, EvalCase("t", expected={"a": "x"}))), {"a"})


class TestRunSuite(unittest.TestCase):
    def _factory(self):
        questions = {"c": choice("pick", ["a", "b"])}

        def resolve(ev, st):
            return [Action("done", terminal=True)]

        def make():
            return Runner(MockClient(rules={"c": {"a": ["good"]}}),
                          Policy(questions, resolvers=[resolve]),
                          ToolRegistry(), StateBuilder(), max_turns=2)
        return make

    def test_outcomes_attached_and_accuracy(self):
        cases = [
            EvalCase("good stuff", expected={"c": "a"}, outcome="success"),
            EvalCase("good things", expected={"c": "a"}, outcome="success"),
            EvalCase("other stuff", expected={"c": "b"}, outcome="failure"),
        ]
        report = run_suite(self._factory(), cases)
        self.assertEqual(len(report.records), 3)  # one terminal turn per case
        self.assertEqual([r.outcome for r in report.records],
                         ["success", "success", "failure"])
        # "other stuff" falls back to options[0]="a", so {c: b} grades False
        self.assertAlmostEqual(report.accuracy, 2 / 3)
        self.assertAlmostEqual(report.case_accuracy, 2 / 3)
        self.assertGreaterEqual(report.ece, 0.0)

    def test_tuner_recovers_known_good_threshold(self):
        # 4 high-confidence successes (0.9) + 1 low-confidence failure (0.5):
        # auto=0.9 acts on exactly the successes.
        cases = ([EvalCase(f"good {i}", expected={"c": "a"}, outcome="success")
                  for i in range(4)]
                 + [EvalCase("other", expected={"c": "a"}, outcome="failure")])
        report = run_suite(self._factory(), cases)
        tuning = sweep_thresholds(report.records, error_budget=0.05)
        self.assertEqual(tuning.best_auto, 0.9)
        self.assertAlmostEqual(tuning.coverage, 0.8)
        self.assertAlmostEqual(tuning.error_rate, 0.0)
        self.assertTrue(tuning.table)  # full grid reported

    def test_tuner_no_pair_under_budget_falls_back(self):
        records = [TurnRecord(turn=1,
                              decisions={"c": {"value": "a", "confidence": 0.9,
                                               "probabilities": {}}},
                              gate="act", actions=[], note="", outcome="failure")]
        tuning = sweep_thresholds(records, error_budget=0.0)
        # every acting cell errs: report the lowest-error (here, max-coverage)
        # acting pair rather than the degenerate act-on-nothing corner
        self.assertEqual(tuning.error_rate, 1.0)
        self.assertEqual(tuning.coverage, 1.0)
        self.assertEqual(tuning.best_auto, 0.5)

    def test_load_cases_jsonl(self):
        import json
        import tempfile

        payload = {"task": "t", "fields": {"x": 1}, "expected": {"c": "a"},
                   "outcome": "success", "expect_run": "completed"}
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(json.dumps(payload) + "\n")
            path = fh.name
        try:
            cases = load_cases(path)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].task, "t")
            self.assertEqual(cases[0].fields, {"x": 1})
            self.assertEqual(cases[0].expect_run, "completed")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
