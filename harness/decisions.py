"""Core data types for System One questions and decisions.

System One models are decision functions: they map (state, questions) to typed
decisions with calibrated probabilities. This module defines those types,
mirroring TypeSafe's three primitives (Choice, Score, Noul) and the wire format
described at https://docs.typesafe.ai/api.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass
class Question:
    """A typed question to ask a System One model about some state."""
    type: QuestionType
    instructions: str = ""
    options: list[str] = field(default_factory=list)  # CHOICE
    levels: list[str] = field(default_factory=list)    # SCORE
    criteria: Any = None  # optional raw criteria for the API (dict / list)

    def __post_init__(self) -> None:
        if self.type is QuestionType.CHOICE and not self.options:
            raise ValueError("CHOICE questions require options")
        if self.type is QuestionType.SCORE and not self.levels:
            raise ValueError("SCORE questions require levels")


def choice(instructions: str, options: list[str] | dict[str, Any]) -> Question:
    """Pick one option from a fixed set.

    ``options`` may be a list of option names, or a dict mapping option ->
    description (or None) that becomes the API's ``criteria``.
    """
    if isinstance(options, dict):
        return Question(QuestionType.CHOICE, instructions=instructions,
                        options=list(options.keys()), criteria=dict(options))
    return Question(QuestionType.CHOICE, instructions=instructions, options=list(options))


def score(instructions: str, levels: list[str]) -> Question:
    """Rate the state on an ordered rubric (an array of level descriptions)."""
    return Question(QuestionType.SCORE, instructions=instructions, levels=list(levels))


def noul(instructions: str, criteria: dict[str, str] | None = None) -> Question:
    """A yes/no claim; the model returns the probability the answer is yes."""
    return Question(QuestionType.NOUL, instructions=instructions, criteria=criteria)


@dataclass
class Decision:
    question: str
    type: QuestionType
    value: Any  # str (choice) | float (score) | bool (noul)
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    @property
    def label(self) -> str:
        return str(self.value)


@dataclass
class Evaluation:
    decisions: dict[str, Decision] = field(default_factory=dict)

    def __getitem__(self, question: str) -> Decision:
        return self.decisions[question]

    def get(self, question: str) -> Decision | None:
        return self.decisions.get(question)

    def min_confidence(self) -> float:
        if not self.decisions:
            return 0.0
        return min(d.confidence for d in self.decisions.values())
