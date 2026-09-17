"""System One model adapters.

The real model is reachable over TypeSafe's API (https://docs.typesafe.ai/api);
a deterministic mock is provided so the harness can be developed and tested
with no network access.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Protocol

from .decisions import Decision, Evaluation, Question, QuestionType


def _read_env_file(path: str = ".env") -> None:
    """Load a gitignored .env into os.environ (stdlib-only dotenv subset)."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


class SystemOneClient(Protocol):
    """Minimal contract for a System One decision model."""
    def evaluate(self, state: str, questions: dict[str, Question]) -> Evaluation: ...


class TypeSafeClient:
    """Adapter for TypeSafe's hosted System One model (Jev).

    POST https://api.typesafe.ai/v1/systemone  {"model": ..., "state": ...,
    "questions": {...}}  ->  typed decisions with probabilities + confidence.

    Reads TYPESAFE_API_KEY from the environment or a local ``.env`` file.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-latest",  # prefer a pinned released version over the moving alias
        endpoint: str = "https://api.typesafe.ai/v1/systemone",
        timeout: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        _read_env_file()
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries

    def evaluate(self, state: str, questions: dict[str, Question]) -> Evaluation:
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is not set (env or .env)")
        import requests

        payload = {
            "model": self.model,
            "state": state,
            "questions": {name: self._question_payload(q) for name, q in questions.items()},
        }

        last_status: int | None = None
        for attempt in range(self.max_retries + 1):
            resp = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            last_status = resp.status_code
            if resp.status_code in (429, 529) and attempt < self.max_retries:
                time.sleep(2 ** attempt)  # exponential backoff
                continue
            if resp.status_code == 401:
                raise PermissionError("TypeSafe API returned 401 — check TYPESAFE_API_KEY")
            resp.raise_for_status()
            return self._parse(resp.json(), questions)

        raise RuntimeError(f"TypeSafe API still failing after retries (status {last_status})")

    @staticmethod
    def _question_payload(q: Question) -> dict[str, Any]:
        if q.type is QuestionType.CHOICE:
            criteria = q.criteria if q.criteria is not None else {o: None for o in q.options}
            return {"type": "choice", "instructions": q.instructions, "criteria": criteria}
        if q.type is QuestionType.SCORE:
            return {"type": "score", "instructions": q.instructions,
                    "criteria": q.criteria if q.criteria is not None else q.levels}
        payload = {"type": "noul", "instructions": q.instructions}
        if q.criteria is not None:
            payload["criteria"] = q.criteria
        return payload

    @staticmethod
    def _parse(data: dict, questions: dict[str, Question]) -> Evaluation:
        """Normalize the response into Evaluation (per https://docs.typesafe.ai/api)."""
        answers = data.get("answers", {}) or {}
        decisions: dict[str, Decision] = {}
        for name, q in questions.items():
            item = answers.get(name) or {}
            if q.type is QuestionType.NOUL:
                # Noul returns a 0..1 probability and no separate confidence field,
                # so confidence is approximated from P(yes). Treat Noul confidence
                # as less principled than Choice/Score confidence when gating.
                prob = float(item.get("noul", 0.0))
                decisions[name] = Decision(
                    name, q.type, prob >= 0.5,
                    probabilities={"true": prob, "false": 1.0 - prob},
                    confidence=prob,
                )
            elif q.type is QuestionType.SCORE:
                decisions[name] = Decision(
                    name, q.type, float(item.get("score", 0.0)),
                    probabilities=item.get("probabilities", {}) or {},
                    confidence=float(item.get("confidence", 0.5)),
                )
            else:  # CHOICE
                decisions[name] = Decision(
                    name, q.type, item.get("choice"),
                    probabilities=item.get("probabilities", {}) or {},
                    confidence=float(item.get("confidence", 0.5)),
                )
        return Evaluation(decisions)


class MockClient:
    """Deterministic keyword-triggered stand-in for local dev and tests.

    ``rules`` is keyed by question name. Each rule may be:
      - a callable ``(state_text, question) -> Decision``
      - a dict mapping an answer to trigger keywords, e.g. ``{"yes": [...], "no": [...]}``
        for Noul, or ``{"billing": ["refund", "invoice"], ...}`` for Choice
      - a literal value
    """

    def __init__(self, rules: dict[str, Any] | None = None) -> None:
        self.rules = rules or {}

    def evaluate(self, state: str, questions: dict[str, Question]) -> Evaluation:
        return Evaluation({name: self._apply(name, q, state) for name, q in questions.items()})

    def _apply(self, name: str, q: Question, state: str) -> Decision:
        rule = self.rules.get(name)
        if callable(rule):
            return rule(state, q)
        if q.type is QuestionType.CHOICE:
            return self._choice(name, q, state, rule)
        if q.type is QuestionType.NOUL:
            return self._noul(name, q, state, rule)
        return Decision(name, q.type, q.upper if hasattr(q, "upper") else 0.0,
                        probabilities={}, confidence=0.5)

    def _choice(self, name: str, q: Question, state: str, rule: Any) -> Decision:
        if isinstance(rule, dict):
            for option, triggers in rule.items():
                if self._hit(state, triggers):
                    return Decision(name, q.type, option, probabilities=self._probs(q, option),
                                    confidence=0.9)
            return Decision(name, q.type, q.options[0], probabilities=self._probs(q, q.options[0]),
                            confidence=0.5)
        selected = rule if rule in q.options else q.options[0]
        return Decision(name, q.type, selected, probabilities=self._probs(q, selected), confidence=0.7)

    def _noul(self, name: str, q: Question, state: str, rule: Any) -> Decision:
        if isinstance(rule, dict):
            if self._hit(state, rule.get("yes")):
                return Decision(name, q.type, True, probabilities={"true": 0.9, "false": 0.1},
                                confidence=0.9)
            if self._hit(state, rule.get("no")):
                return Decision(name, q.type, False, probabilities={"true": 0.1, "false": 0.9},
                                confidence=0.9)
        return Decision(name, q.type, False, probabilities={"true": 0.3, "false": 0.7},
                        confidence=0.6)

    @staticmethod
    def _hit(state: str, triggers: Any) -> bool:
        if triggers is None:
            return False
        if isinstance(triggers, str):
            triggers = [triggers]
        low = state.lower()
        return any(str(t).lower() in low for t in triggers)

    @staticmethod
    def _probs(q: Question, selected: str) -> dict[str, float]:
        n = max(1, len(q.options) - 1)
        return {o: (0.9 if o == selected else 0.1 / n) for o in q.options}
