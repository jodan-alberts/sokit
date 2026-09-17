"""The Runner: the decision-driven agent loop.

Owns everything the System One model can't do on its own: multi-turn iteration,
tool execution, datasource refresh, and termination guarantees (budgets,
no-progress detection, escalation).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .client import SystemOneClient
from .confidence import ConfidenceGate, Gate
from .context import StateBuilder
from .decisions import Evaluation, noul
from .generate import TextGenerator
from .memory import LongTermMemory
from .policy import Action, Policy
from .state import State
from .telemetry import Telemetry
from .tools import ToolRegistry, render


class _Escalate(Exception):
    """Internal control flow: abort the run with an escalation + note."""
    def __init__(self, note: str) -> None:
        super().__init__(note)
        self.note = note


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
        generator: TextGenerator | None = None,
        max_turns: int = 8,
        max_tool_calls: int = 20,
        no_progress_limit: int = 3,
        max_tool_errors: int = 3,
        on_confirm: Callable[[Evaluation, State], bool] | None = None,
        on_turn: Callable[[int, Evaluation, Gate, list[Action]], None] | None = None,
    ) -> None:
        self.client = client
        self.policy = policy
        self.tools = tools
        self.state_builder = state_builder
        self.confidence_gate = confidence_gate or ConfidenceGate()
        self.longterm_memory = longterm_memory
        self.generator = generator
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.no_progress_limit = no_progress_limit
        self.max_tool_errors = max_tool_errors
        self.on_confirm = on_confirm
        self.on_turn = on_turn
        self.telemetry = Telemetry()

    def run(self, task: str, fields: dict | None = None) -> RunResult:
        state = State(task=task, fields=fields or {})
        final: Evaluation | None = None
        tool_calls = 0
        consecutive_errors = 0
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
            if self.on_turn is not None:
                # Streaming hook for UIs (fire-and-forget: never affects control flow).
                self.on_turn(turn, evaluation, gate, actions)

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
                    try:
                        action = self._maybe_generate(action, evaluation, state)
                    except _Escalate as esc:
                        self.telemetry.record(turn, evaluation, gate, actions, esc.note)
                        return RunResult("escalated", evaluation, state, self.telemetry)
                    result = self.tools.execute(action, evaluation, state)
                    tool_calls += 1
                    verdict = "OK" if result.ok else "ERROR"
                    state.add_event("tool", action.tool, f"{verdict}: {result.output}")
                    if result.ok:
                        consecutive_errors = 0
                    elif result.output.startswith("skipped (already ran)"):
                        # Idempotency guard, not a tool failure: neutral for the
                        # error counter (neither increments nor resets it).
                        pass
                    else:
                        consecutive_errors += 1
                        if consecutive_errors >= self.max_tool_errors:
                            note = (f"{consecutive_errors} consecutive tool errors — escalating")
                            self.telemetry.record(turn, evaluation, gate, actions, note)
                            return RunResult("escalated", evaluation, state, self.telemetry)
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

    def _maybe_generate(self, action: Action, evaluation: Evaluation, state: State) -> Action:
        """Generate-then-validate (§10, pattern B).

        Renders the action's prompt template, drafts text via the generator,
        then re-evaluates with a validator Noul gated by the confidence gate.
        The draft only ever fills ``args[slot]`` — generator output never
        influences which action was chosen, and the validation verdict never
        re-enters ``policy.resolve()``. Any failure escalates with evidence;
        nothing proceeds silently.
        """
        if action.generate is None:
            return action
        spec = action.generate
        slot = spec.get("slot")
        if not slot:
            raise _Escalate("generate spec missing 'slot' — escalating")
        if self.generator is None:
            state.add_event("note", action.tool or "generate",
                            "no generator configured for generate action")
            raise _Escalate("no generator configured — escalating")

        prompt = render(spec.get("prompt", ""), evaluation, state)
        try:
            draft = self.generator.generate(
                prompt, {"evaluation": evaluation, "state": state})
        except Exception as exc:  # noqa: BLE001 — generator failure escalates, never proceeds
            state.add_event("generator", action.tool or "generate", f"ERROR: {exc}")
            raise _Escalate(f"generator failed ({exc}) — escalating") from exc
        state.add_event("draft", action.tool or "generate", str(draft))

        vname = spec.get("validator_question", "draft_ok")
        vinstructions = spec.get("validator", "Is this draft safe, correct, and on-policy?")
        validation = self.client.evaluate(
            self.state_builder.assemble(state), {vname: noul(vinstructions)})
        verdict = validation.get(vname)
        state.add_event(
            "validator", vname,
            f"{verdict.value if verdict else '?'} "
            f"(conf {verdict.confidence:.2f})" if verdict else "no verdict",
        )
        # Same thresholds as the main gate, scoped to the validator question
        # (the main gate is often restricted to routing questions, which are
        # absent here — gating on those would always escalate).
        vgate = ConfidenceGate(
            auto=self.confidence_gate.auto, escalate=self.confidence_gate.escalate,
            questions=[vname],
        ).gate(validation)
        if verdict is None or bool(verdict.value) is not True:
            raise _Escalate(f"validator rejected draft ({vname}={getattr(verdict, 'value', '?')}) — escalating")
        if vgate is Gate.ESCALATE:
            raise _Escalate("validator low confidence — escalating")
        if vgate is Gate.CONFIRM and self.on_confirm is not None:
            if not self.on_confirm(validation, state):
                raise _Escalate("declined at validator confirm gate")
        return replace(action, args={**action.args, slot: draft})

    @staticmethod
    def _fingerprint(evaluation: Evaluation, state: State) -> str:
        decisions = tuple(sorted((q, str(d.value)) for q, d in evaluation.decisions.items()))
        events = tuple(e.content for e in state.events)
        return f"{decisions}|{events}"
