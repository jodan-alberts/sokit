"""Tests for tool hardening: exceptions, timeouts, retries, cache semantics."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import (
    Action,
    FunctionTool,
    MockClient,
    Policy,
    Runner,
    StateBuilder,
    ToolRegistry,
    ToolResult,
    choice,
)
from harness.decisions import Evaluation
from harness.state import State


def _eval_state():
    return Evaluation({}), State(task="t")


class TestToolHardening(unittest.TestCase):
    def test_raising_tool_becomes_error_result(self):
        def boom(args, ctx):
            raise RuntimeError("kaput")

        reg = ToolRegistry().register(FunctionTool("boom", boom))
        ev, st = _eval_state()
        result = reg.execute(Action("b", tool="boom"), ev, st)
        self.assertFalse(result.ok)
        self.assertIn("ERROR", result.output)
        self.assertIn("kaput", result.output)

    def test_timeout_tool(self):
        def slow(args, ctx):
            time.sleep(2)
            return ToolResult("slow", True, "late")

        reg = ToolRegistry(default_timeout=0.1).register(FunctionTool("slow", slow))
        ev, st = _eval_state()
        result = reg.execute(Action("s", tool="slow"), ev, st)
        self.assertFalse(result.ok)
        self.assertIn("timeout", result.output)

    def test_per_tool_timeout_override(self):
        def slow(args, ctx):
            time.sleep(0.3)
            return ToolResult("slow", True, "late")

        reg = ToolRegistry(default_timeout=5.0).register(
            FunctionTool("slow", slow, timeout=0.05)
        )
        ev, st = _eval_state()
        result = reg.execute(Action("s", tool="slow"), ev, st)
        self.assertFalse(result.ok)
        self.assertIn("timeout", result.output)

    def test_retry_succeeds_on_flaky_tool(self):
        calls = []

        def flaky(args, ctx):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("flaky")
            return ToolResult("flaky", True, "recovered")

        reg = ToolRegistry(default_retries=3).register(FunctionTool("flaky", flaky))
        ev, st = _eval_state()
        result = reg.execute(Action("f", tool="flaky"), ev, st)
        self.assertTrue(result.ok)
        self.assertEqual(len(calls), 3)

    def test_ok_false_not_retried(self):
        calls = []

        def nope(args, ctx):
            calls.append(1)
            return ToolResult("nope", False, "business failure")

        reg = ToolRegistry(default_retries=3).register(FunctionTool("nope", nope))
        ev, st = _eval_state()
        result = reg.execute(Action("n", tool="nope"), ev, st)
        self.assertFalse(result.ok)
        self.assertEqual(len(calls), 1)

    def test_failed_call_not_cached(self):
        calls = []

        def sometimes(args, ctx):
            calls.append(1)
            if len(calls) == 1:
                return ToolResult("t", False, "first fails")
            return ToolResult("t", True, "second works")

        reg = ToolRegistry().register(FunctionTool("t", sometimes))
        ev, st = _eval_state()
        action = Action("t", tool="t", args={"x": "1"})
        first = reg.execute(action, ev, st)
        second = reg.execute(action, ev, st)
        self.assertFalse(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(len(calls), 2)

    def test_success_still_cached(self):
        calls = []

        def once(args, ctx):
            calls.append(1)
            return ToolResult("t", True, "ok")

        reg = ToolRegistry().register(FunctionTool("once", once))
        ev, st = _eval_state()
        action = Action("t", tool="once")
        reg.execute(action, ev, st)
        second = reg.execute(action, ev, st)
        self.assertFalse(second.ok)
        self.assertIn("already ran", second.output)
        self.assertEqual(len(calls), 1)

    def test_runner_escalates_after_consecutive_tool_errors(self):
        questions = {"next": choice("next step", ["go"])}

        def resolve(ev, st):
            return [Action("run", tool="bad")]

        def bad(args, ctx):
            raise RuntimeError("always fails")

        tools = ToolRegistry().register(FunctionTool("bad", bad))
        client = MockClient(rules={"next": "go"})
        runner = Runner(client, Policy(questions, resolvers=[resolve]), tools,
                        StateBuilder(), max_turns=8, max_tool_errors=3)
        result = runner.run("go", {})
        self.assertEqual(result.outcome, "escalated")
        self.assertTrue(any("consecutive tool errors" in r.note for r in result.telemetry.records))
        # errors are ERROR events, run completes (no exception propagates)
        self.assertTrue(any("ERROR" in e.content for e in result.state.events))

    def test_runner_error_counter_resets_on_success(self):
        questions = {"next": choice("next step", ["go"])}
        calls = []

        def sometimes(args, ctx):
            calls.append(1)
            if len(calls) % 2 == 1:
                raise RuntimeError("odd fails")
            return ToolResult("t", True, "even works")

        def resolve(ev, st):
            return [Action("run", tool="t", args={"n": "{{task}}", "c": str(len(calls))},
                           allow_repeat=True)]

        tools = ToolRegistry().register(FunctionTool("t", sometimes))
        client = MockClient(rules={"next": "go"})
        runner = Runner(client, Policy(questions, resolvers=[resolve]), tools,
                        StateBuilder(), max_turns=4, max_tool_errors=3)
        result = runner.run("go", {})
        # alternating fail/ok with unique args never hits 3 consecutive errors
        self.assertNotEqual(result.outcome, "escalated")


if __name__ == "__main__":
    unittest.main()
