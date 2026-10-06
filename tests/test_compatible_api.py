"""Offline transport tests: subprocess.run is mocked before every test."""

import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from customer_policy_verifier.generation import compatible_api as api


SCHEMA = {"json_schema": {"schema": {
    "type": "object", "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"], "additionalProperties": False,
}}}


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.run = self.enterContext(patch.object(api.subprocess, "run"))
        self.run.side_effect = AssertionError("Unexpected transport call in offline test")
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.directory / "request.json"
        self.config = api.load_config()
        self.key = "dummy-test-secret-never-a-real-key"
        self.body = {field: self.config[field] for field in (
            "model", "temperature", "max_tokens", "response_format", "reasoning", "provider")}
        self.body["messages"] = [{"role": "user", "content": "Return JSON"}]
        self.path.write_text(json.dumps(self.body))
        self.gate = api.RequestGate(8)

    def invoke(self):
        return api.call(self.config, self.key, self.body, SCHEMA, self.path, self.gate)

    def response(self, status=200, payload=None, headers=""):
        if payload is None:
            payload = {"id": "test", "model": api.MODEL,
                       "choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.00001}}
        def run(command, **kwargs):
            Path(command[command.index("--dump-header") + 1]).write_text(headers)
            return subprocess.CompletedProcess(command, 0, json.dumps(payload) + f"\n{status}", "")
        return run

    def test_config_needs_no_credentials_or_network(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("No credential reads")):
            self.assertEqual(api.load_config()["model"], "qwen/qwen3.7-flash")
        self.run.assert_not_called()

    def test_credentials_only_on_stdin_and_headers_removed(self):
        self.run.side_effect = self.response()
        output, metadata = self.invoke()
        command = self.run.call_args.args[0]
        options = self.run.call_args.kwargs
        self.assertEqual(output, {"ok": True})
        self.assertEqual(metadata["reported_cost_usd"], 0.00001)
        self.assertNotIn(self.key, " ".join(command))
        self.assertIn("Authorization: Bearer " + self.key, options["input"])
        self.assertNotIn(self.key, self.path.read_text())
        self.assertEqual(command[1], "--disable")
        self.assertNotIn("--location", command)
        self.assertFalse(Path(command[command.index("--dump-header") + 1]).exists())

    def test_untrusted_endpoint_rejected_before_curl(self):
        self.config["api_base"] = "https://api.kunyou.asia/v1"
        with self.assertRaisesRegex(ValueError, "only be sent"):
            self.invoke()
        self.run.assert_not_called()

    def test_saved_body_or_frozen_settings_mismatch_rejected(self):
        self.path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Saved request file"):
            self.invoke()
        self.path.write_text(json.dumps(self.body))
        self.body["max_tokens"] = 999
        with self.assertRaisesRegex(ValueError, "frozen configuration"):
            self.invoke()
        self.run.assert_not_called()

    def test_429_retries_and_reduces_concurrency(self):
        replies = iter([
            self.response(429, {"error": {"message": "rate limited"}}, "Retry-After: 0\r\n"),
            self.response(),
        ])
        self.run.side_effect = lambda *args, **kwargs: next(replies)(*args, **kwargs)
        _, metadata = self.invoke()
        self.assertEqual(len(metadata["attempts"]), 2)
        self.assertEqual(self.gate.limit, 4)
        self.assertEqual(self.gate.rate_limit_events, 1)
        self.assertEqual(self.gate.active, 0)

    def test_401_is_not_retried_and_redacts_key(self):
        self.run.side_effect = self.response(401, {"error": {"message": self.key}})
        with self.assertRaises(api.ApiFailure) as caught:
            self.invoke()
        self.assertEqual(self.run.call_count, 1)
        self.assertEqual(self.gate.active, 0)
        self.assertNotIn(self.key, str(caught.exception.attempts))
        self.assertIn("[REDACTED]", str(caught.exception.attempts))

    def test_bad_model_json_is_saved_for_rejection_without_retry(self):
        self.run.side_effect = self.response(payload={"choices": [{
            "finish_reason": "stop", "message": {"content": "not JSON"}}]})
        output, metadata = self.invoke()
        self.assertEqual(output, {"unparsed_content": "not JSON"})
        self.assertIn("model output is not valid JSON", metadata["schema_issues"])
        self.assertIsNone(metadata["reported_cost_usd"])
        self.assertEqual(self.run.call_count, 1)

    def test_truncated_or_invalid_schema_is_rejected(self):
        self.run.side_effect = self.response(payload={"choices": [{
            "finish_reason": "length", "message": {"content": '{"ok": "wrong type"}'}}]})
        _, metadata = self.invoke()
        self.assertIn("finish_reason=length", metadata["schema_issues"])
        self.assertIn("$.ok should be boolean", metadata["schema_issues"])

    def test_http_200_transient_error_is_retried(self):
        replies = iter([
            self.response(payload={"error": {"code": 503, "message": "upstream failure"}}),
            self.response(),
        ])
        self.run.side_effect = lambda *args, **kwargs: next(replies)(*args, **kwargs)
        with patch.object(self.gate.stop, "wait", return_value=False):
            _, metadata = self.invoke()
        self.assertEqual(metadata["attempts"][0]["http_status"], 200)
        self.assertEqual(metadata["attempts"][0]["api_error_code"], 503)
        self.assertEqual(self.run.call_count, 2)

    def test_http_200_auth_error_is_not_retried(self):
        self.run.side_effect = self.response(payload={"error": {"code": "401", "message": "bad key"}})
        with self.assertRaises(api.ApiFailure):
            self.invoke()
        self.assertEqual(self.run.call_count, 1)

    def test_http_200_missing_choices_is_transport_failure(self):
        self.run.side_effect = self.response(payload={"unexpected": True})
        with self.assertRaisesRegex(api.ApiFailure, "not a Chat Completions result"):
            self.invoke()
        self.assertEqual(self.run.call_count, 1)

    def test_timeout_retries_are_bounded_and_cleanup_runs(self):
        self.run.side_effect = subprocess.TimeoutExpired("curl", 190)
        with patch.object(self.gate.stop, "wait", return_value=False):
            with self.assertRaises(api.ApiFailure) as caught:
                self.invoke()
        self.assertEqual(len(caught.exception.attempts), 4)
        self.assertTrue(all(a["curl_exit"] == 28 for a in caught.exception.attempts))
        self.assertEqual(self.run.call_count, 4)
        self.assertEqual(self.gate.active, 0)
        for call in self.run.call_args_list:
            command = call.args[0]
            self.assertFalse(Path(command[command.index("--dump-header") + 1]).exists())

    def test_stop_prevents_requests(self):
        self.gate.stop.set()
        with self.assertRaises(InterruptedError):
            self.invoke()
        self.run.assert_not_called()

    def test_gate_never_exceeds_limit(self):
        gate = api.RequestGate(2)
        observed = []
        lock = threading.Lock()
        def work(_):
            gate.acquire()
            try:
                with lock:
                    observed.append(gate.active)
                time.sleep(0.01)
            finally:
                gate.release()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(work, range(8)))
        self.assertLessEqual(max(observed), 2)
        self.assertEqual(gate.active, 0)


if __name__ == "__main__":
    unittest.main()
