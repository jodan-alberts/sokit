"""Tests for the interactive CLI (mock-backed, deterministic)."""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import cli, tui
from harness import AnthropicGenerator, HttpGenerator


def run_cli(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(list(argv))
    return code, buf.getvalue()


class TestTui(unittest.TestCase):
    def test_style_plain_by_default_under_test(self):
        self.assertEqual(tui.style("hi", fg=tui.RED, bold=True), "hi")

    def test_style_forced_color(self):
        out = tui.style("hi", fg=tui.RED, bold=True, force=True)
        self.assertIn("\x1b[", out)
        self.assertTrue(out.endswith(tui.RESET))

    def test_no_color_env_disables(self):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertEqual(tui.style("hi", fg=tui.RED, force=None), "hi")

    def test_conf_bar_shape(self):
        bar = tui.conf_bar(0.5, width=4)
        self.assertIn("██░░", bar)
        self.assertIn("0.50", bar)

    def test_panel_borders(self):
        out = tui.panel("t", ["line one", "longer line two"])
        self.assertIn("╭", out)
        self.assertIn("╰", out)
        self.assertIn("line one", out)

    def test_gate_chips(self):
        for gate in ("act", "confirm", "escalate"):
            self.assertIn(gate, tui.gate_chip(gate))

    def test_spinner_disabled_is_noop(self):
        with tui.Spinner("x", enabled=False) as spin:
            self.assertIsNotNone(spin)


class FakeResp:
    """Minimal urlopen context manager returning canned JSON."""
    last_request = None

    def __init__(self, payload):
        import json as _json
        self._body = _json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


class TestGenerators(unittest.TestCase):
    def _patch_urlopen(self, payload):
        from unittest.mock import patch as _patch

        def fake(req, timeout=None):
            FakeResp.last_request = req
            return FakeResp(payload)

        return _patch("urllib.request.urlopen", side_effect=fake)

    def test_anthropic_text_blocks_joined(self):
        gen = AnthropicGenerator(model="claude-x", api_key="sk-test")
        payload = {"content": [{"type": "text", "text": "Hello."},
                               {"type": "tool_use", "id": "1"},
                               {"type": "text", "text": "Kind regards"}]}
        with self._patch_urlopen(payload):
            self.assertEqual(gen.generate("hi"), "Hello.\nKind regards")
        req = FakeResp.last_request
        self.assertTrue(req.full_url.endswith("/v1/messages"))
        self.assertEqual(req.get_header("X-api-key"), "sk-test")
        self.assertEqual(req.get_header("Anthropic-version"), "2023-06-01")

    def test_anthropic_needs_model_and_key(self):
        with self.assertRaises(RuntimeError):
            AnthropicGenerator(model="").generate("hi", {})
        import os as _os
        from unittest.mock import patch as _patch
        with _patch.dict(_os.environ, {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                AnthropicGenerator(model="claude-x").generate("hi")

    def test_anthropic_no_text_blocks_errors(self):
        gen = AnthropicGenerator(model="claude-x", api_key="sk-test")
        with self._patch_urlopen({"content": [{"type": "tool_use", "id": "1"}]}):
            with self.assertRaises(RuntimeError):
                gen.generate("hi")

    def test_http_extra_headers_merged(self):
        gen = HttpGenerator(model="m", api_key="k",
                            extra_headers={"HTTP-Referer": "https://x.test"})
        payload = {"choices": [{"message": {"content": "draft"}}]}
        with self._patch_urlopen(payload):
            self.assertEqual(gen.generate("hi"), "draft")
        req = FakeResp.last_request
        self.assertTrue(req.full_url.endswith("/chat/completions"))
        self.assertEqual(req.get_header("Http-referer"), "https://x.test")
        self.assertEqual(req.get_header("Authorization"), "Bearer k")


class TestCli(unittest.TestCase):
    def test_list_shows_all_agents(self):
        code, out = run_cli(["list"])
        self.assertEqual(code, 0)
        for name in ("support", "incident", "draft"):
            self.assertIn(name, out)

    def test_run_support_completes(self):
        code, out = run_cli(["run", "support", "I want a refund",
                             "--fields", '{"account_id": "acct_123"}', "--mock"])
        self.assertEqual(code, 0)
        self.assertIn("outcome: completed", out)
        self.assertIn("refund issued", out)

    def test_run_incident_completes(self):
        code, out = run_cli(["run", "incident", "pager", "--mock", "--quiet"])
        self.assertEqual(code, 0)
        self.assertIn("outcome: completed", out)

    def test_run_draft_completes(self):
        code, out = run_cli(["run", "draft", "I want a refund",
                             "--fields", '{"account_id": "acct_123"}', "--mock"])
        self.assertEqual(code, 0)
        self.assertIn("outcome: completed", out)
        self.assertIn("message sent", out)

    def test_run_unknown_agent(self):
        code, _ = run_cli(["run", "nope", "task", "--mock"])
        self.assertEqual(code, 2)

    def test_run_bad_fields(self):
        code, _ = run_cli(["run", "support", "task", "--fields", "{oops", "--mock"])
        self.assertEqual(code, 2)

    def test_run_non_object_fields(self):
        code, _ = run_cli(["run", "support", "task", "--fields", "[1]", "--mock"])
        self.assertEqual(code, 2)

    def test_gate_override_escalates(self):
        # Impossibly strict auto threshold: nothing auto-acts, decline confirms.
        with patch("builtins.input", return_value="n"):
            code, out = run_cli(["run", "support", "I want a refund",
                                 "--fields", '{"account_id": "acct_123"}',
                                 "--mock", "--auto", "0.999", "--quiet"])
        self.assertEqual(code, 1)
        self.assertIn("outcome: escalated", out)

    def test_telemetry_out_written(self):
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
            path = fh.name
        try:
            code, _ = run_cli(["run", "support", "I want a refund",
                               "--fields", '{"account_id": "acct_123"}',
                               "--mock", "--quiet", "--telemetry-out", path])
            self.assertEqual(code, 0)
            with open(path, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            self.assertTrue(records)
            self.assertIn("decisions", records[0])
        finally:
            os.remove(path)

    def test_on_turn_hook_fires(self):
        from harness import Runner

        seen = []
        client = cli.AGENTS["support"]["mock_client"]()
        runner = cli.build_runner("support", client, (0.8, 0.5), 6,
                                  on_turn=lambda t, e, g, a: seen.append(t))
        result = runner.run("I want a refund", {"account_id": "acct_123"})
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(seen, [1, 2, 3])

    def test_repl_quit(self):
        with patch("builtins.input", side_effect=[":quit"]):
            code, _ = run_cli(["repl", "support", "--mock"])
        self.assertEqual(code, 0)

    def test_repl_default_agent_is_incident(self):
        with patch("builtins.input", side_effect=[":quit"]):
            code, out = run_cli(["repl", "--mock"])
        self.assertEqual(code, 0)
        self.assertIn("incident", out)

    def test_generator_openai_without_key_fails_fast(self):
        import os as _os
        from unittest.mock import patch as _patch

        with _patch.dict(_os.environ, {"OPENAI_API_KEY": ""}):
            code, _ = run_cli(["run", "draft", "task", "--mock",
                               "--generator", "openai"])
        self.assertEqual(code, 2)

    def test_generator_anthropic_without_key_fails_fast(self):
        import os as _os
        from unittest.mock import patch as _patch

        with _patch.dict(_os.environ, {"ANTHROPIC_API_KEY": ""}):
            code, _ = run_cli(["run", "draft", "task", "--mock",
                               "--generator", "anthropic",
                               "--generator-model", "claude-x"])
        self.assertEqual(code, 2)

    def test_generator_anthropic_needs_model(self):
        import os as _os
        from unittest.mock import patch as _patch

        with _patch.dict(_os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            code, _ = run_cli(["run", "draft", "task", "--mock",
                               "--generator", "anthropic"])
        self.assertEqual(code, 2)

    def test_repl_task_then_quit(self):
        with patch("builtins.input",
                   side_effect=["I want a refund", ":quit"]):
            code, out = run_cli(["repl", "support", "--mock"])
        self.assertEqual(code, 0)
        self.assertIn("outcome: completed", out)

    def test_repl_commands(self):
        script = [":fields {\"account_id\": \"acct_999\"}",
                  ":gates 0.7 0.4",
                  ":trace off",
                  ":bogus",
                  ":agent draft",
                  ":quit"]
        with patch("builtins.input", side_effect=script):
            code, out = run_cli(["repl", "support", "--mock"])
        self.assertEqual(code, 0)
        self.assertIn("switched to draft", out)
        self.assertIn("unknown command", out)


if __name__ == "__main__":
    unittest.main()
