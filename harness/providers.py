"""Concrete ContextProviders.

A provider runs before each evaluate() and injects external data into the state.
The model never holds credentials or touches I/O; it can only trigger these
pre-declared providers/tools.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Callable

from .context import Document
from .memory import LongTermMemory
from .state import State


@dataclass
class FunctionProvider:
    """Wrap any ``(State) -> list[Document]`` callable as a provider."""
    name: str
    fn: Callable[[State], list[Document]]

    def gather(self, state: State) -> list[Document]:
        return self.fn(state)


@dataclass
class FilesProvider:
    name: str = "files"
    path: str = ""

    def gather(self, state: State) -> list[Document]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                return [Document("files", fh.read())]
        except OSError as exc:
            return [Document("files", f"[unreadable: {exc}]")]


@dataclass
class HttpProvider:
    name: str = "http"
    url: str = ""
    headers: dict | None = None
    timeout: float = 5.0

    def gather(self, state: State) -> list[Document]:
        import urllib.request

        req = urllib.request.Request(self.url, headers=self.headers or {})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return [Document("http", resp.read().decode("utf-8", "replace"))]


@dataclass
class ClockProvider:
    name: str = "clock"

    def gather(self, state: State) -> list[Document]:
        return [Document("clock", _dt.datetime.now().isoformat(timespec="seconds"))]


@dataclass
class MemoryProvider:
    """Retrieves from long-term memory each turn (naive keyword search)."""
    name: str = "memory"
    memory: LongTermMemory | None = None
    k: int = 5

    def gather(self, state: State) -> list[Document]:
        if self.memory is None:
            return []
        hits = self.memory.search(state.task, k=self.k)
        return [Document("memory", h) for h in hits]


class SqlProvider:
    """Stub: wire a DB driver here. The model can only trigger this provider
    through pre-declared actions; credentials stay in the harness."""
    def __init__(self, name: str = "sql", query: str = "") -> None:
        self.name = name
        self.query = query

    def gather(self, state: State) -> list[Document]:
        raise NotImplementedError("wire a database connection here")


class WebSearchProvider:
    """Stub: plug a search API here (the model decides what to search via a
    routing Choice; the harness executes the query)."""
    def __init__(self, name: str = "web") -> None:
        self.name = name

    def gather(self, state: State) -> list[Document]:
        raise NotImplementedError("wire a web-search API here")
