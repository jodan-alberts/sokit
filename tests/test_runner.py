"""Tests for the Runner loop."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import (
    Action,
    ConfidenceGate,
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


def _runner(rules, questions, resolve, tools=None, providers=()):
    client = MockClient(rules=rules)
    state_builder = StateBuilder(providers=list(providers))
    return Runner(client, Policy(questions, resolvers=[resolve]),
                  tools or ToolRegistry(), state_builder, max_turns=4)


class TestRunner(unittest.TestCase):
    def test_completes_via_terminal_action(self):
        questions = {"next": choice("next step", ["act", "done"])}

        def resolve(ev, st):
            if ev["next"].value == "done":
                return [Action("finish", terminal=True)]
            return [Action("work", tool="noop")]

        tools = ToolRegistry().register(FunctionTool("noop", lambda a, c: ToolResult("noop", True, "ok")))
        runner = _runner({"next": {"done": ["done"], "act": ["go"]}}, questions, resolve, tools=tools)
        result = runner.run("please go and then be done", {})
        self.assertEqual(result.outcome, "completed")
        self.assertGreaterEqual(len(result.telemetry.records), 1)

    def test_escalates_on_low_confidence(self):
        questions = {"next": choice("next step", ["a", "b"])}

        def resolve(ev, st):
            return [Action("x", terminal=True)]

        runner = _runner({"next": {}}, questions, resolve)
        runner.confidence_gate = ConfidenceGate(auto=0.95, escalate=0.9)
        result = runner.run("something", {})
        self.assertEqual(result.outcome, "escalated")

    def test_budget_exhausted_without_terminal(self):
        questions = {"next": choice("next step", ["loop"])}

        def resolve(ev, st):
            return [Action("loop", tool="noop")]

        tools = ToolRegistry().register(FunctionTool("noop", lambda a, c: ToolResult("noop", True, "ok")))
        runner = _runner({"next": {"loop": ["loop"]}}, questions, resolve, tools=tools)
        result = runner.run("loop", {})
        self.assertEqual(result.outcome, "budget_exhausted")

    def test_no_progress_detection(self):
        questions = {"next": choice("next step", ["loop"])}

        def resolve(ev, st):
            return [Action("loop")]  # a no-op action: no tool, not terminal

        runner = _runner({"next": {"loop": ["loop"]}}, questions, resolve)
        runner.no_progress_limit = 3
        result = runner.run("loop", {})
        self.assertEqual(result.outcome, "no_progress")

    def test_tool_result_recorded_and_idempotent(self):
        questions = {"go": noul("go?")}

        def resolve(ev, st):
            return [Action("run", tool="echo", args={"msg": "hello"}), Action("stop", terminal=True)]

        tools = ToolRegistry().register(
            FunctionTool("echo", lambda a, c: ToolResult("echo", True, a.get("msg", ""), data=a.get("msg", "")))
        )
        memory = InMemoryStore()
        runner = _runner({"go": {"yes": ["go"]}}, questions, resolve, tools=tools)
        runner.longterm_memory = memory
        result = runner.run("go", {})
        self.assertEqual(result.outcome, "completed")
        self.assertTrue(any("hello" in e.content for e in result.state.events))
        self.assertEqual(memory.items[0]["content"], "hello")

    def test_idempotency_guard_blocks_repeat(self):
        questions = {"next": choice("next step", ["loop"])}

        def resolve(ev, st):
            return [Action("run", tool="echo", args={"msg": "x"})]

        calls = []
        tools = ToolRegistry().register(FunctionTool("echo", lambda a, c: calls.append(1) or ToolResult("echo", True, "ok")))
        runner = _runner({"next": {"loop": ["loop"]}}, questions, resolve, tools=tools)
        result = runner.run("loop", {})
        # max_turns=4, but idempotency means the tool actually runs only once
        self.assertEqual(len(calls), 1)
        self.assertIn(result.outcome, ("budget_exhausted", "no_progress"))


if __name__ == "__main__":
    unittest.main()
