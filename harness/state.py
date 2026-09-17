"""Working memory for a harness run.

A System One model has no hidden state, so all continuity lives here: the task,
structured fields, prior decisions, and the log of tool events that gets written
back into the state the model reads each turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    kind: str       # "tool" | "note" | ...
    source: str     # tool name / origin
    content: str
    turn: int


@dataclass
class State:
    task: str
    fields: dict[str, Any] = field(default_factory=dict)      # structured program state
    events: list[Event] = field(default_factory=list)         # tool/observation log
    decisions: list[dict[str, Any]] = field(default_factory=list)  # prior decisions
    turn: int = 0

    def add_event(self, kind: str, source: str, content: str) -> None:
        self.events.append(Event(kind=kind, source=source, content=content, turn=self.turn))
