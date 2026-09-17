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
    """Read-only SQLite datasource (stdlib ``sqlite3``).

    ``query`` supports the tool-arg template style (``{{fields.x}}`` /
    ``{{task}}``); ``params`` are bound separately via DB-API placeholders
    and never interpolated. Read-only is enforced two ways: the file is
    opened with ``mode=ro`` and only ``SELECT``/``WITH`` statement prefixes
    are accepted. Output is one Document with rows as compact JSON lines,
    truncated at ``max_rows`` with a truncation marker. DB errors degrade to
    a ``[sql error: …]`` document — providers must never crash the loop.

    Pass ``connection`` (e.g. a shared ``:memory:`` connection) instead of a
    file path for tests; otherwise ``db_path`` is opened per gather.
    """
    def __init__(
        self,
        db_path: str = "",
        query: str = "",
        params: list | tuple | None = None,
        max_rows: int = 50,
        read_only: bool = True,
        name: str = "sql",
        connection: Any = None,
    ) -> None:
        self.db_path = db_path
        self.query = query
        self.params = list(params) if params is not None else []
        self.max_rows = max_rows
        self.read_only = read_only
        self.name = name
        self.connection = connection

    def gather(self, state: State) -> list[Document]:
        import re as _re
        import sqlite3 as _sqlite3

        rendered = self._render(self.query, state)
        first = _re.sub(r"^\s*(--[^\n]*\n|\s|/\*.*?\*/)*", "", rendered,
                        flags=_re.DOTALL).strip().upper()
        if not (first.startswith("SELECT") or first.startswith("WITH")):
            return [Document(self.name, "[sql error: only SELECT/WITH queries are allowed]")]
        try:
            if self.connection is not None:
                cursor = self.connection.execute(rendered, self.params)
                rows = cursor.fetchall()
                columns = [str(d[0]) for d in (cursor.description or [])]
                return [self._format(rows, columns)]
            if self.db_path == ":memory:":
                raise _sqlite3.OperationalError("no shared connection for :memory:")
            if self.read_only:
                uri = f"file:{self.db_path}?mode=ro"
                conn = _sqlite3.connect(uri, uri=True)
            else:
                conn = _sqlite3.connect(self.db_path)
            try:
                conn.row_factory = _sqlite3.Row
                cursor = conn.execute(rendered, self.params)
                rows = cursor.fetchall()
                columns = [str(d[0]) for d in (cursor.description or [])]
                return [self._format(rows, columns)]
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 — graceful degradation, like FilesProvider
            return [Document(self.name, f"[sql error: {exc}]")]

    def _format(self, rows: list, columns: list[str]) -> Document:
        import json as _json

        total = len(rows)
        lines = []
        for row in rows[: self.max_rows]:
            if isinstance(row, dict):
                mapping = row
            elif hasattr(row, "keys"):  # sqlite3.Row
                mapping = {col: row[col] for col in columns}
            else:  # plain tuple from a cursor without row_factory
                mapping = {col: row[i] for i, col in enumerate(columns)}
            lines.append(_json.dumps(mapping, separators=(",", ":"), default=str))
        text = "\n".join(lines)
        if total > self.max_rows:
            marker = f"…(truncated, {total} total)"
            text = (text + "\n" + marker) if text else marker
        return Document(self.name, text)

    @staticmethod
    def _render(template: str, state: State) -> str:
        def repl(match: Any) -> str:
            key = match.group(1).strip()
            if key == "task":
                return state.task
            if key.startswith("fields."):
                return str(state.fields.get(key.split(".", 1)[1], ""))
            return match.group(0)

        import re as _re
        return _re.sub(r"\{\{([^{}]+)\}\}", repl, template)


class WebSearchProvider:
    """Stub: plug a search API here (the model decides what to search via a
    routing Choice; the harness executes the query)."""
    def __init__(self, name: str = "web") -> None:
        self.name = name

    def gather(self, state: State) -> list[Document]:
        raise NotImplementedError("wire a web-search API here")
