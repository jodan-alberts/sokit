"""Long-term memory."""
from __future__ import annotations

import json
import os
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


def _keyword_search(items: list[dict], query: str, k: int = 5) -> list[str]:
    """Shared keyword-overlap scoring (same ranking as InMemoryStore)."""
    q = set(query.lower().split())
    scored: list[tuple[int, str]] = []
    for item in items:
        words = set(item["content"].lower().split())
        score = len(q & words)
        if score:
            scored.append((score, item["content"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [content for _, content in scored[:k]]


class JsonlStore:
    """Crash-tolerant file-backed long-term memory (stdlib only).

    Append-per-add: every ``add()`` is flushed to disk immediately, and the
    full file is loaded on init, so a crash loses nothing acknowledged.
    Scoring matches :class:`InMemoryStore` (keyword overlap).

    Limits: no file locking — single-writer only. Concurrent writers may
    interleave lines (each line is still valid JSON, but ordering/count can
    surprise).     Vector backends remain a documented future, not this class.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.items: list[dict] = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(entry, dict) and "content" in entry:
                        entry.setdefault("metadata", {})
                        self.items.append(entry)

    def add(self, content: str, metadata: dict | None = None) -> None:
        entry = {"content": content, "metadata": metadata or {}}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        self.items.append(entry)

    def search(self, query: str, k: int = 5) -> list[str]:
        return _keyword_search(self.items, query, k)
