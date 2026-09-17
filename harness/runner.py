"""The Runner: the decision-driven agent loop.

Owns everything the System One model can't do on its own: multi-turn iteration,
tool execution, datasource refresh, and termination guarantees (budgets,
no-progress detection, escalation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .client import SystemOneClient
from .confidence import ConfidenceGate, Gate
from .context import StateBuilder
from .decisions import Evaluation
from .memory import LongTermMemory
from .policy import Action, Policy
from .state import State
from .telemetry import Telemetry
from .tools import ToolRegistry


@dataclass
class RunResult:
    outcome: str                      # "completed" | "escalated" | "budget_exhausted" | "no_progress"
    evaluation: Evaluation | None
    state: State
    telemetry: Telemetry


class Runner:
    def __init__(
        self,
        client: SystemOneClient,
        policy: Policy,
        tools: ToolRegistry,
        state_builder: StateBuilder,
        confidence_gate: ConfidenceGate | None = None,
        longterm_memory: LongTermMemory | None = None,
        max_turns: int = 8,
        max_tool_calls: int = 20,
        no_progress_limit: int = 3,
        on_confirm: Callable[[Evaluation, State], bool] | None = None,
    ) -> None:
        self.client = client
        self.policy = policy
        self.tools = tools
        self.state_builder = state_builder
        self.confidence_gate = confidence_gate or ConfidenceGate()
        self.longterm_memory = longterm_memory
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.no_progress_limit = no_progress_limit
        self.on_confirm = on_confirm
        self.telemetry = Telemetry()

    def run(self, task: str, fields: dict | None = None) -> RunResult:
        state = State(task=task, fields=fields or {})
        final: Evaluation | None = None
        tool_calls = 0
        last_fingerprint: str | None = None
        no_progress = 0

        for turn in range(1, self.max_turns + 1):
            state.turn = turn
            questions = self.policy.questions_for(state)
            assembled = self.state_builder.assemble(state)
            evaluation = self.client.evaluate(assembled, questions)
            final = evaluation

            # Write decisions back into state so the model "remembers" them.
            for name, decision in evaluation.decisions.items():
                state.decisions.append({
                    "turn": turn,
                    "question": name,
                    "value": decision.value,
                    "confidence": decision.confidence,
                })

            gate = self.confidence_gate.gate(evaluation)
            if gate is Gate.ESCALATE:
                self.telemetry.record(turn, evaluation, gate, [], "escalated by confidence gate")
                return RunResult("escalated", evaluation, state, self.telemetry)

            if gate is Gate.CONFIRM and self.on_confirm is not None:
                if not self.on_confirm(evaluation, state):
                    self.telemetry.record(turn, evaluation, gate, [], "declined at confirm gate")
                    return RunResult("escalated", evaluation, state, self.telemetry)

            actions = self.policy.resolve(evaluation, state)
            note = "auto-approved" if gate is Gate.CONFIRM else ""

            terminal = False
            escalate = False
            for action in actions:
                if action.terminal:
                    terminal = True
                    escalate = action.escalate
                    break
                if action.tool is not None:
                    if tool_calls >= self.max_tool_calls:
                        self.telemetry.record(turn, evaluation, gate, actions, "tool-call budget exhausted")
                        return RunResult("escalated", evaluation, state, self.telemetry)
                    result = self.tools.execute(action, evaluation, state)
                    tool_calls += 1
                    verdict = "OK" if result.ok else "ERROR"
                    state.add_event("tool", action.tool, f"{verdict}: {result.output}")
                    if self.longterm_memory is not None and result.ok and result.data is not None:
                        self.longterm_memory.add(str(result.data))

            # No-progress detection: identical (decisions, events) fingerprint
            # for N consecutive turns means the loop is stuck -> escalate.
            fingerprint = self._fingerprint(evaluation, state)
            no_progress = no_progress + 1 if fingerprint == last_fingerprint else 0
            last_fingerprint = fingerprint
            if no_progress >= self.no_progress_limit:
                self.telemetry.record(turn, evaluation, gate, actions, "no progress detected")
                return RunResult("no_progress", evaluation, state, self.telemetry)

            self.telemetry.record(turn, evaluation, gate, actions, note)

            if terminal:
                return RunResult("escalated" if escalate else "completed", evaluation, state, self.telemetry)

        return RunResult("budget_exhausted", final, state, self.telemetry)

    @staticmethod
    def _fingerprint(evaluation: Evaluation, state: State) -> str:
        decisions = tuple(sorted((q, str(d.value)) for q, d in evaluation.decisions.items()))
        events = tuple(e.content for e in state.events)
        return f"{decisions}|{events}"
