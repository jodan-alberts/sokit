"""Tests for Phase 2 datasources: SqlProvider + JsonlStore (stdlib only)."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import JsonlStore, SqlProvider
from harness.state import State


def _memdb():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE accounts (id TEXT PRIMARY KEY, plan TEXT, refundable INTEGER)")
    conn.executemany(
        "INSERT INTO accounts (id, plan, refundable) VALUES (?, ?, ?)",
        [("acct_123", "pro", 1), ("acct_999", "basic", 0), ("acct_007", "pro", 1)],
    )
    conn.commit()
    return conn


class TestSqlProvider(unittest.TestCase):
    def test_select_works(self):
        conn = _memdb()
        try:
            p = SqlProvider(query="SELECT id, plan FROM accounts WHERE id = 'acct_123'",
                            connection=conn)
            docs = p.gather(State(task="t"))
            self.assertEqual(len(docs), 1)
            row = json.loads(docs[0].content)
            self.assertEqual(row, {"id": "acct_123", "plan": "pro"})
        finally:
            conn.close()

    def test_insert_rejected(self):
        conn = _memdb()
        try:
            p = SqlProvider(query="INSERT INTO accounts VALUES ('x', 'pro', 1)",
                            connection=conn)
            docs = p.gather(State(task="t"))
            self.assertIn("sql error", docs[0].content)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0], 3)
        finally:
            conn.close()

    def test_row_cap_and_truncation_marker(self):
        conn = _memdb()
        try:
            p = SqlProvider(query="SELECT id FROM accounts", max_rows=2, connection=conn)
            docs = p.gather(State(task="t"))
            lines = docs[0].content.splitlines()
            self.assertEqual(len(lines), 3)  # 2 rows + marker
            self.assertIn("truncated", lines[-1])
            self.assertIn("3 total", lines[-1])
        finally:
            conn.close()

    def test_template_rendering(self):
        conn = _memdb()
        try:
            p = SqlProvider(query="SELECT plan FROM accounts WHERE id = '{{fields.account_id}}'",
                            connection=conn)
            docs = p.gather(State(task="t", fields={"account_id": "acct_999"}))
            self.assertEqual(json.loads(docs[0].content), {"plan": "basic"})
        finally:
            conn.close()

    def test_params_bound_separately(self):
        conn = _memdb()
        try:
            p = SqlProvider(query="SELECT id FROM accounts WHERE plan = ?",
                            params=["pro"], connection=conn)
            docs = p.gather(State(task="t"))
            self.assertEqual(len(docs[0].content.splitlines()), 2)
        finally:
            conn.close()

    def test_error_doc_on_bad_query(self):
        conn = _memdb()
        try:
            p = SqlProvider(query="SELECT nope FROM missing", connection=conn)
            docs = p.gather(State(task="t"))
            self.assertIn("[sql error:", docs[0].content)
        finally:
            conn.close()

    def test_file_read_only(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            path = fh.name
        try:
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE t (a TEXT)")
            conn.execute("INSERT INTO t VALUES ('hi')")
            conn.commit()
            conn.close()
            # Writes through the provider are rejected (statement gate).
            p = SqlProvider(db_path=path, query="DELETE FROM t")
            self.assertIn("sql error", p.gather(State(task="t"))[0].content)
            # Reads work through the read-only open.
            p = SqlProvider(db_path=path, query="SELECT a FROM t")
            docs = p.gather(State(task="t"))
            self.assertEqual(json.loads(docs[0].content), {"a": "hi"})
            # Direct writes with mode=ro fail at the sqlite layer too.
            with self.assertRaises(sqlite3.OperationalError):
                ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                ro.execute("INSERT INTO t VALUES ('x')")
        finally:
            os.remove(path)


class TestJsonlStore(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
            path = fh.name
        os.remove(path)  # start absent: store must handle missing file
        try:
            store = JsonlStore(path)
            store.add("refund issued for acct_123", {"kind": "refund"})
            store.add("checkout deploy 4821 rolled back")
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)

            reopened = JsonlStore(path)
            self.assertEqual(len(reopened.items), 2)
            hits = reopened.search("refund acct_123")
            self.assertEqual(hits, ["refund issued for acct_123"])
            # same keyword scoring as InMemoryStore: ranking by overlap
            hits = reopened.search("deploy checkout rolled")
            self.assertEqual(hits, ["checkout deploy 4821 rolled back"])
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_search_matches_inmemory_ranking(self):
        from harness import InMemoryStore

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
            path = fh.name
        try:
            mem = InMemoryStore()
            disk = JsonlStore(path)
            for text in ["alpha beta gamma", "alpha beta", "alpha"]:
                mem.add(text)
                disk.add(text)
            self.assertEqual(disk.search("alpha beta gamma"), mem.search("alpha beta gamma"))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
