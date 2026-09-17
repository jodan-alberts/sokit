"""The policy: how decisions map to actions, and which questions to ask.

This is the "program" of the harness. Because a System One model can't emit
tool-call arguments, every action must be pre-enumerated here; arguments come
from state or templates, never from model-generated text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .decisions import Evaluation, Question
from .state import State


@dataclass
class Action:
    name: str
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)  # templates: {{fields.x}}, {{decision.y}}, {{task}}
    terminal: bool = False
    escalate: bool = False       # a terminal action that hands off to a human
    allow_repeat: bool = False  # default: side-effecting tools are idempotent-guarded


RouteResolver = Callable[[Evaluation, State], list[Action] | None]
QuestionSelector = Callable[[State], dict[str, Question]]


def choice_route(question: str, value: str, actions: list[Action]) -> RouteResolver:
    """Route when a Choice decision equals ``value``."""
    def _resolve(evaluation: Evaluation, state: State) -> list[Action] | None:
        d = evaluation.get(question)
        if d is not None and d.value == value:
            return actions
        return None
    return _resolve


def noul_route(question: str, value: bool, actions: list[Action]) -> RouteResolver:
    """Route when a Noul decision is ``value``."""
    def _resolve(evaluation: Evaluation, state: State) -> list[Action] | None:
        d = evaluation.get(question)
        if d is not None and bool(d.value) is value:
            return actions
        return None
    return _resolve


@dataclass
class Policy:
    questions: dict[str, Question]
    resolvers: list[RouteResolver] = field(default_factory=list)
    default: list[Action] = field(default_factory=list)
    select_questions: QuestionSelector | None = None  # two-pass question selection (§5)
    version: str = "1"                                # version your decision schema

    def questions_for(self, state: State) -> dict[str, Question]:
        if self.select_questions is not None:
            return self.select_questions(state)
        return self.questions

    def resolve(self, evaluation: Evaluation, state: State) -> list[Action]:
        for resolver in self.resolvers:
            actions = resolver(evaluation, state)
            if actions:
                return actions
        return self.default
