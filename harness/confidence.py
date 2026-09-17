"""Confidence-gated control flow (TypeSafe's recommended pattern)."""
from __future__ import annotations

from enum import Enum

from .decisions import Evaluation


class Gate(str, Enum):
    ACT = "act"
    CONFIRM = "confirm"
    ESCALATE = "escalate"


class ConfidenceGate:
    """Map an evaluation's confidence to a control-flow verdict.

    Thresholds live in *your* code and must be tuned on your own labeled data —
    TypeSafe explicitly says confidence thresholds are use-case-specific.

    ``questions`` restricts gating to the decisions that actually drive control
    flow (e.g. the routing ``next_action`` choice). Speculative/informational
    questions (like a root-cause ``hypothesis``) are often *honestly* low-
    confidence early on and should not force an escalation. Defaults to all.
    """

    def __init__(self, auto: float = 0.8, escalate: float = 0.5,
                 questions: list[str] | None = None) -> None:
        if not (0.0 <= escalate <= auto <= 1.0):
            raise ValueError("require 0 <= escalate <= auto <= 1")
        self.auto = auto
        self.escalate = escalate
        self.questions = questions

    def gate(self, evaluation: Evaluation) -> Gate:
        if self.questions:
            confidences = [d.confidence for q in self.questions
                           if (d := evaluation.get(q)) is not None]
        else:
            confidences = [d.confidence for d in evaluation.decisions.values()]
        confidence = min(confidences) if confidences else 0.0
        if confidence >= self.auto:
            return Gate.ACT
        if confidence < self.escalate:
            return Gate.ESCALATE
        return Gate.CONFIRM
