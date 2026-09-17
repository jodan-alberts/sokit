"""State assembly and the ContextProvider datasource protocol."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .state import State


@dataclass
class Document:
    source: str
    content: str


class ContextProvider(Protocol):
    name: str

    def gather(self, state: State) -> list[Document]: ...


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


@dataclass
class StateBuilder:
    """Renders State + external context into the single text/blob the model reads.

    Budget accounting is explicit because the ~32k-token request budget is the
    binding constraint, not model intelligence.
    """
    max_chars: int = 24000
    providers: list[Any] = field(default_factory=list)  # ContextProvider or callable(state)->docs
    max_prior_decisions: int = 12
    max_history_events: int = 12

    def assemble(self, state: State) -> str:
        parts = [f"# Task\n{state.task}"]

        # Structured fields are kept separate and injected compactly, not as prose.
        if state.fields:
            parts.append("# Structured fields\n" + json.dumps(state.fields, default=str))

        docs: list[Document] = []
        for provider in self.providers:
            try:
                found = self._gather(provider, state)
            except Exception as exc:  # a provider failure must not kill the loop
                found = [Document(getattr(provider, "name", "provider"), f"[error: {exc}]")]
            if isinstance(found, Document):
                found = [found]
            docs.extend(found)
        if docs:
            rendered = "\n\n".join(f"[{d.source}]\n{d.content}" for d in docs)
            parts.append("# External context\n" + rendered)

        if state.decisions:
            history = "\n".join(
                f"- turn {d['turn']}: {d['question']} = {d['value']} (conf {d['confidence']:.2f})"
                for d in state.decisions[-self.max_prior_decisions:]
            )
            parts.append("# Prior decisions\n" + history)

        if state.events:
            events = "\n".join(f"- [{e.source}] {e.content}" for e in state.events[-self.max_history_events:])
            parts.append("# Tool history\n" + events)

        return truncate("\n\n".join(parts), self.max_chars)

    @staticmethod
    def _gather(provider: Any, state: State) -> Any:
        return provider(state) if callable(provider) else provider.gather(state)
