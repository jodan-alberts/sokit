"""Decision + outcome logging.

Every turn is recorded with the full probability distribution (not just the
argmax), the gate, and the actions taken. Outcome labels are attached offline;
they are what make the confidence gate real (see calibration.py).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .confidence import Gate
from .decisions import Evaluation
from .policy import Action


@dataclass
class TurnRecord:
    turn: int
    decisions: dict[str, dict[str, Any]]
    gate: str
    actions: list[dict[str, Any]]
    note: str = ""
    outcome: str | None = None  # ground-truth label, e.g. "success" / "failure"


class Telemetry:
    def __init__(self) -> None:
        self.records: list[TurnRecord] = []

    def record(
        self,
        turn: int,
        evaluation: Evaluation,
        gate: Gate,
        actions: list[Action],
        note: str = "",
    ) -> None:
        decisions = {
            q: {
                "value": d.value,
                "confidence": d.confidence,
                "probabilities": d.probabilities,
            }
            for q, d in evaluation.decisions.items()
        }
        self.records.append(
            TurnRecord(
                turn=turn,
                decisions=decisions,
                gate=gate.value,
                actions=[asdict(a) for a in actions],
                note=note,
            )
        )

    def attach_outcome(self, turn: int, outcome: str) -> None:
        for record in self.records:
            if record.turn == turn:
                record.outcome = outcome
                return
        raise ValueError(f"no record for turn {turn}")

    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as fh:
            for record in self.records:
                fh.write(json.dumps(asdict(record), default=str) + "\n")
