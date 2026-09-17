"""Long-term memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class LongTermMemory(Protocol):
    def add(self, content: str, metadata: dict | None = None) -> None: ...
    def search(self, query: str, k: int = 5) -> list[str]: ...


@dataclass
class InMemoryStore:
    """Trivial keyword-overlap long-term memory for demos/tests.

    Swap for a vector DB in production.
    """
    items: list[dict] = field(default_factory=list)

    def add(self, content: str, metadata: dict | None = None) -> None:
        self.items.append({"content": content, "metadata": metadata or {}})

    def search(self, query: str, k: int = 5) -> list[str]:
        q = set(query.lower().split())
        scored: list[tuple[int, str]] = []
        for item in self.items:
            words = set(item["content"].lower().split())
            score = len(q & words)
            if score:
                scored.append((score, item["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in scored[:k]]
