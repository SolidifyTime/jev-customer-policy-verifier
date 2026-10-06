"""Offline lifecycle tests: no network and no real credentials."""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from customer_policy_verifier.common.paths import resolve_path
from customer_policy_verifier.generation import bulk_generate as bulk
from customer_policy_verifier.generation import compatible_api
from customer_policy_verifier.review import export_candidates


def row(group=1):
    return {"orig_conv_id": group, "intent": "verify", "instruction": "Ask for the account ID.",
            "conversation": ["turn 1: customer: I need help with my account."],
            "orig_gt_response": "turn 2: agent: What is your account ID?",
            "manipulated_gt_response": "turn 2: agent: I have verified your account."}


class FakeAPI:
    def __init__(self, fail_at=None, reject_labels=()):
        self.calls = []
        self.fail_at, self.reject_labels = fail_at, reject_labels

    def __call__(self, config, key, body, schema, request_path, gate):
        self.calls.append(request_path.name)
        if len(self.calls) == self.fail_at:
            raise compatible_api.ApiFailure("simulated transport failure", [{"http_status": 503}])
        payload = json.loads(body["messages"][1]["content"])
        if schema["json_schema"]["name"] == "scene_extraction":
            # Fixture evidence is exactly quoted from the original input.
            output = {"rules": [{"quote": "Ask for the account ID.", "meaning": "Request ID"}],
                      "customer_request": "Account help", "known_facts": [{"turn_index": 0,
                      "quote": "I need help with my account.", "meaning": "Needs help"}],
                      "unknowns": ["account ID"], "dialogue_progress": "ID not provided",
                      "response_claims": ["Response action"], "label_rationale": "Fixture",
                      "can_expand": payload.get("expected_label") not in self.reject_labels, "issues": []}
        else:
            slot = int(request_path.name.split("-")[1])
            output = {"conversation": [f"turn 1: customer: Please help with my account, wording {slot}."],
                      "candidate_response": f"turn 2: agent: May I have your account ID, variant {slot}?",
                      "generation_note": "offline fixture"}
        return output, {"model": config["model"], "requested_model": config["model"],
                        "schema_issues": [], "usage": {"prompt_tokens": 100, "completion_tokens": 40, "cost": 0.001},
                        "reported_cost_usd": 0.001, "attempts": [{"http_status": 200}]}


class BulkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw, self.split = self.root / "raw.json", self.root / "split.json"
        bulk.atomic_json(self.raw, [row(), row(2), row(3)])
        bulk.atomic_json(self.split, {"source_sha256": bulk.digest(self.raw), "seed": 42,
                                    "group_assignment": {"1": "train", "2": "validation", "3": "test"}})
        self.manifest = self.root / "manifest.json"
        bulk.atomic_json(self.manifest, {"source_file": str(self.raw), "source_sha256": bulk.digest(self.raw),
                          "split_file": str(self.split), "split_sha256": bulk.digest(self.split)})
        self.selection = bulk.build_selection(self.manifest, 10)
        self.run = self.root / "run"

    def tearDown(self):
        self.tmp.cleanup()

    def execute(self, fake, max_sources=None, budget=10):
        batch = bulk.Batch(self.run, self.selection, 2, budget, "mock-secret", call=fake)
        with contextlib.redirect_stdout(io.StringIO()):
            result = batch.execute(max_sources)
        return result

    def test_both_labels_only_train_and_ten_variants(self):
        self.assertEqual({s["expected_label"] for s in self.selection["samples"]}, {"compliant", "violating"})
        self.assertEqual({s["orig_conv_id"] for s in self.selection["samples"]}, {1})
        fake = FakeAPI()
        result = self.execute(fake)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed_generations"], 20)
        self.assertEqual(len(fake.calls), 22)
        self.assertAlmostEqual(result["usage"]["reported_cost_usd"], .022)
        self.assertEqual(self.execute(fake)["status"], "completed")
        self.assertEqual(len(fake.calls), 22, "completed resume must make zero calls")

    def test_partial_resume_reuses_saved_stages(self):
        fake = FakeAPI()
        result = self.execute(fake, max_sources=1)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(fake.calls), 11)
        self.assertEqual(self.execute(fake)["completed_generations"], 20)
        self.assertEqual(len(fake.calls), 22)

    def test_generated_bulk_is_accepted_by_exporter_and_spot_check_import(self):
        self.selection["samples"] = self.selection["samples"][:1]
        bulk.atomic_json(self.run / "selection.json", self.selection)
        self.assertEqual(self.execute(FakeAPI())["status"], "completed")
        output = self.root / "export"
        with contextlib.redirect_stdout(io.StringIO()):
            export_candidates.prepare([self.run], output, 3, 42, self.root / "no-exclusions.jsonl")
        summary = bulk.read_json(output / "summary.json")
        self.assertEqual(summary["candidate_count"], 10)
        self.assertEqual(summary["spot_check_count"], 3)
        review = output / "spot_check.md"
        review.write_text(review.read_text().replace("：待填写", "：通过"))
        with contextlib.redirect_stdout(io.StringIO()):
            export_candidates.apply_spot_check(output, "assistant")
        self.assertEqual(bulk.read_json(output / "summary.json")["exported_count"], 10)

    def test_interrupted_source_resumes_without_repeating_saved_outputs(self):
        fake = FakeAPI(fail_at=5)
        first = self.execute(fake, max_sources=1)
        self.assertEqual(first["status"], "paused")
        self.assertEqual(first["usage"]["saved_api_responses"], 4)
        resumed = FakeAPI()
        second = self.execute(resumed)
        self.assertEqual(second["status"], "completed")
        self.assertEqual(len(resumed.calls), 18)
        self.assertEqual(second["usage"]["saved_api_responses"], 22)

    def test_rejected_extraction_never_generates(self):
        fake = FakeAPI(reject_labels=("compliant", "violating"))
        result = self.execute(fake)
        self.assertEqual(result["completed_count"], 2)
        self.assertEqual(result["completed_generations"], 0)
        self.assertEqual(len(fake.calls), 2)

    def test_budget_stops_and_missing_cost_is_not_zero(self):
        result = self.execute(FakeAPI(), budget=.001, max_sources=1)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["usage"]["saved_api_responses"], 1)
        self.assertEqual(result["completed_generations"], 0)
        path = bulk.source_directory(self.run, 0) / "extract.json"
        saved = bulk.read_json(path)
        saved["api"]["reported_cost_usd"] = None
        bulk.atomic_json(path, saved)
        fake = FakeAPI()
        result = self.execute(fake)
        self.assertEqual(result["usage"]["missing_cost_count"], 1)
        self.assertIsNone(result["usage"]["reported_cost_usd"])
        self.assertEqual(fake.calls, [])

    def test_cached_request_edit_blocks_resume(self):
        self.execute(FakeAPI(fail_at=3), max_sources=1)
        request = bulk.source_directory(self.run, 0) / "extract-request.json"
        body = bulk.read_json(request)
        body["temperature"] = 1
        bulk.atomic_json(request, body)
        fake = FakeAPI()
        result = self.execute(fake, max_sources=1)
        self.assertEqual(result["status"], "paused")
        self.assertIn("Cached request differs", result["stop_reason"])
        self.assertEqual(fake.calls, [])

    def test_missing_cost_on_final_response_still_pauses(self):
        self.selection["samples"] = self.selection["samples"][:1]
        fake = FakeAPI()
        def final_cost_missing(*args):
            output, api = fake(*args)
            if len(fake.calls) == 11:
                api["reported_cost_usd"] = None
            return output, api
        result = self.execute(final_cost_missing)
        self.assertEqual(result["completed_count"], 1)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["usage"]["missing_cost_count"], 1)

    def test_prepare_does_not_read_credentials_or_call_api(self):
        argv = ["bulk_generate.py", "--dataset-manifest", str(self.manifest), "--run-dir", str(self.run)]
        with patch.object(sys, "argv", argv), patch.object(bulk, "load_key", side_effect=AssertionError("key read")), \
                patch.object(compatible_api, "call", side_effect=AssertionError("network")), \
                contextlib.redirect_stdout(io.StringIO()):
            bulk.main()
        self.assertEqual(bulk.read_json(self.run / "summary.json")["status"], "prepared_awaiting_user_confirmation")
        self.assertFalse((self.run / "sources").exists())

    def test_source_or_split_mutation_blocks_plan(self):
        body = bulk.read_json(self.split)
        body["group_assignment"]["1"] = "test"
        bulk.atomic_json(self.split, body)
        with self.assertRaisesRegex(ValueError, "split changed"):
            bulk.build_selection(self.manifest, 10)

    def test_relocated_manifest_sources_keep_frozen_bytes_and_hashes(self):
        legacy_raw, legacy_split = self.root / "legacy/raw.json", self.root / "legacy/split.json"
        bulk.atomic_json(self.root / "configs/layout-migration.json", {"path_migrations": {
            "legacy/raw.json": "raw.json", "legacy/split.json": "split.json"}})
        manifest = bulk.read_json(self.manifest)
        manifest.update(source_file=str(legacy_raw), split_file=str(legacy_split))
        bulk.atomic_json(self.manifest, manifest)
        frozen_bytes = self.manifest.read_bytes()
        with patch.object(bulk, "resolve_path", side_effect=lambda value: resolve_path(value, root=self.root)), \
                patch.object(bulk, "EXCLUSIONS", self.root / "no-exclusions.jsonl"), \
                patch.object(bulk, "implementation_hashes", return_value=self.selection["implementation_sha256"]):
            selection = bulk.build_selection(self.manifest, 10)
        self.assertEqual(selection["source_file"], str(self.raw.resolve()))
        self.assertEqual(selection["split_file"], str(self.split.resolve()))
        self.assertEqual(selection["source_sha256"], manifest["source_sha256"])
        self.assertEqual(selection["split_sha256"], manifest["split_sha256"])
        self.assertEqual(self.manifest.read_bytes(), frozen_bytes)
        self.assertFalse(legacy_raw.exists())

    def test_changed_implementation_refuses_resume_before_credentials_or_api(self):
        old_selection = copy.deepcopy(self.selection)
        old_selection["implementation_sha256"]["bulk_generate.py"] = "0" * 64
        frozen = self.run / "selection.json"
        bulk.atomic_json(frozen, old_selection)
        frozen_bytes = frozen.read_bytes()
        argv = ["bulk-generate", "--resume", str(self.run), "--execute", "--max-cost-usd", "0.01"]
        with patch.object(sys, "argv", argv), patch.object(bulk, "load_key", side_effect=AssertionError("key read")), \
                patch.object(compatible_api, "call", side_effect=AssertionError("network")), \
                self.assertRaisesRegex(ValueError, "implementation changed; refusing mixed resume"):
            bulk.main()
        self.assertEqual(frozen.read_bytes(), frozen_bytes)
        self.assertFalse((self.run / "execution.json").exists())
        self.assertFalse((self.run / "plan.json").exists())


if __name__ == "__main__":
    unittest.main()
