"""Offline tests for frozen bulk exports and the existing small-run format."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from customer_policy_verifier.review import export_candidates as export


class BulkExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw.json"
        self.split_path = self.root / "split.json"
        self.run = self.root / "run"
        self.run.mkdir()
        self.rows = [{
            "instruction": "Verify identity before cancelling a card.",
            "conversation": ["turn 1: customer: Please cancel my card."],
            "orig_gt_response": "turn 2: agent: May I verify your identity first?",
            "manipulated_gt_response": "turn 2: agent: I cancelled your card already.",
            "orig_conv_id": 100, "intent": "cancel_card",
        }, {
            "instruction": "Verify identity before cancelling a card.",
            "conversation": ["turn 1: customer: Cancel my second card, please."],
            "orig_gt_response": "turn 2: agent: Please verify your identity first.",
            "manipulated_gt_response": "turn 2: agent: Your card is cancelled.",
            "orig_conv_id": 200, "intent": "cancel_card",
        }]
        export.write_json(self.raw, self.rows)
        self.split = {"source_sha256": export.sha256(self.raw),
                      "group_assignment": {"100": "train", "200": "test"}}
        export.write_json(self.split_path, self.split)
        self.samples = [self.sample(0, label) for label in ("compliant", "violating")]
        self.selection = {
            "kind": "bulk", "model": "qwen/qwen3.7-flash",
            "prompt_version": "bulk-v1/pilot-v10/multi-v3",
            "source_file": str(self.raw), "split_file": str(self.split_path),
            "source_sha256": export.sha256(self.raw),
            "split_sha256": export.sha256(self.split_path),
            "samples": self.samples, "variants_per_scene": 10,
        }
        self.summary = {
            "model": self.selection["model"], "prompt_version": self.selection["prompt_version"],
            "status": "completed", "completed_count": 2, "results": [],
        }
        for number, sample in enumerate(self.samples):
            folder = self.run / "sources" / f"{number:05d}"
            folder.mkdir(parents=True)
            self.save_output(folder / "extract.json", self.scene(sample))
            generations = []
            for slot in range(1, 11):
                self.save_output(folder / f"generate-{slot:02d}.json", self.variant(sample, slot))
                generations.append({"variant_number": slot, "status": "candidate_for_human_review", "issues": []})
            self.summary["results"].append({
                "source_number": number, "source_row_index": 0,
                "expected_label": sample["expected_label"], "status": "completed",
                "generations": generations,
            })
        self.persist()

    def sample(self, index, label):
        row = self.rows[index]
        response_key = "orig_gt_response" if label == "compliant" else "manipulated_gt_response"
        return {"source_row_index": index, "expected_label": label, "policy": row["instruction"],
                "conversation": row["conversation"], "candidate_response": row[response_key],
                "orig_conv_id": row["orig_conv_id"], "intent": row["intent"]}

    @staticmethod
    def scene(sample):
        return {"rules": [{"quote": sample["policy"], "meaning": "Identity must be verified."}],
                "customer_request": "Cancel the card.",
                "known_facts": [{"turn_index": 0, "quote": sample["conversation"][0],
                                 "meaning": "The customer requests cancellation."}],
                "unknowns": ["Identity verification."], "dialogue_progress": "Cancellation requested.",
                "response_claims": [sample["candidate_response"]],
                "label_rationale": sample["expected_label"], "can_expand": True, "issues": []}

    @staticmethod
    def variant(sample, slot):
        return {"conversation": [f"turn 1: customer: I would like to cancel my card, please. Wording {slot}."],
                "candidate_response": sample["candidate_response"], "generation_note": "Reworded request."}

    def save_output(self, path, output):
        export.write_json(path, {"output": output, "api": {"schema_issues": []}})

    def persist(self):
        export.write_json(self.run / "selection.json", self.selection)
        export.write_json(self.run / "summary.json", self.summary)

    def records(self):
        return list(export.generation_records(self.run, self.rows, self.split))

    def prepare(self, output):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(export, "known_rejected_generations", return_value=set()):
            export.prepare([self.run], output, 3, 42, self.root / "no-exclusions.jsonl")

    def test_complete_unordered_results_and_both_labels_for_one_row(self):
        self.summary["results"].reverse()
        self.persist()
        records = self.records()
        self.assertEqual(len(records), 20)
        self.assertEqual({r[0]["expected_label"] for r in records}, {"compliant", "violating"})
        self.assertEqual([r[3] for r in records[:10]], list(range(1, 11)))

    def test_three_checks_for_entire_batch_and_apply_uses_frozen_paths(self):
        output = self.root / "export"
        self.prepare(output)
        manifest = export.read_json(output / "manifest.json")
        self.assertEqual(manifest["source_file"], str(self.raw.resolve()))
        self.assertEqual(manifest["split_file"], str(self.split_path.resolve()))
        self.assertEqual(len(manifest["spot_check_ids"]), 3)
        self.assertEqual(export.read_json(output / "summary.json")["candidate_count"], 20)
        review = output / "spot_check.md"
        review.write_text(review.read_text().replace("结论（通过／剔除／待定）：待填写", "结论（通过／剔除／待定）：通过"))
        with contextlib.redirect_stdout(io.StringIO()):
            export.apply_spot_check(output, "human")
        rows = list(map(json.loads, (output / "training_variants.jsonl").read_text().splitlines()))
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row["metadata"]["review_status"] == "spot_checked" for row in rows), 3)

    def test_missing_generation_slot_rejected(self):
        self.summary["results"][0]["generations"].pop()
        self.persist()
        with self.assertRaisesRegex(ValueError, "generation slots"):
            self.records()

    def test_missing_generation_file_rejected(self):
        (self.run / "sources/00000/generate-07.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing generation"):
            self.records()

    def test_duplicate_result_source_number_rejected(self):
        self.summary["results"][1]["source_number"] = 0
        self.persist()
        with self.assertRaisesRegex(ValueError, "duplicate source numbers"):
            self.records()

    def test_duplicate_selection_key_rejected(self):
        self.selection["samples"][1] = copy.deepcopy(self.selection["samples"][0])
        self.persist()
        with self.assertRaisesRegex(ValueError, "duplicate source and label"):
            self.records()

    def test_test_partition_rejected_even_for_rejected_extraction(self):
        self.selection["samples"][0] = self.sample(1, "compliant")
        self.summary["results"][0].update(source_row_index=1, status="rejected_extraction", generations=[])
        self.persist()
        with self.assertRaisesRegex(ValueError, "outside the training split"):
            self.records()

    def test_interrupted_run_rejected(self):
        self.summary["status"] = "interrupted"
        self.persist()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.records()

    def test_rejected_extraction_has_no_generated_records(self):
        self.summary["results"][0].update(status="rejected_extraction", generations=[])
        self.persist()
        self.assertEqual(len(self.records()), 10)

    def test_frozen_source_and_split_hashes_enforced(self):
        for field in ("source_sha256", "split_sha256"):
            original = self.selection[field]
            self.selection[field] = "0" * 64
            self.persist()
            with self.assertRaisesRegex(ValueError, "frozen split differs"):
                self.records()
            self.selection[field] = original

    def test_source_changes_after_export_block_apply(self):
        output = self.root / "export"
        self.prepare(output)
        self.raw.write_text(self.raw.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed since export"):
            export.apply_spot_check(output, "human")

    def test_preflight_rejections_are_excluded_and_hash_bound(self):
        run = self.root / "runs" / "archive" / "bulk-preflight" / "fixture-v1"
        source = run / "sources" / "00000"
        source.mkdir(parents=True)
        generated = source / "generate-03.json"
        self.save_output(generated, self.variant(self.samples[0], 3))
        checksum = export.sha256(generated)
        export.write_json(run / "preflight-review.json", {"decisions": [{
            "verdict": "剔除", "generation_file": "sources/00000/generate-03.json",
            "generation_sha256": checksum, "reason": "Only greeting differs from variant 2"}]})
        with patch.object(export, "ROOT", self.root):
            self.assertEqual(export.known_rejected_generations(), {checksum})
            generated.write_text(generated.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed since"):
                export.known_rejected_generations()

    def test_legacy_completed_batch_and_manifest_fallback(self):
        self.selection = {key: self.selection[key] for key in ("model", "prompt_version", "source_sha256", "samples")}
        self.summary["results"] = [{"source_row_index": 0, "expected_label": sample["expected_label"],
                                    "status": "candidate_for_human_review"} for sample in self.samples]
        self.persist()
        for number, sample in enumerate(self.samples):
            self.save_output(self.run / f"sample-{number}-extract.json", self.scene(sample))
            self.save_output(self.run / f"sample-{number}-generate.json", self.variant(sample, 1))
        self.assertEqual(len(self.records()), 2)
        output = self.root / "legacy-export"
        with patch.object(export, "RAW", self.raw), patch.object(export, "SPLIT", self.split_path):
            self.prepare(output)
            manifest = export.read_json(output / "manifest.json")
            del manifest["source_file"], manifest["split_file"]
            export.write_json(output / "manifest.json", manifest)
            review = output / "spot_check.md"
            review.write_text(review.read_text().replace("结论（通过／剔除／待定）：待填写", "结论（通过／剔除／待定）：通过"))
            with contextlib.redirect_stdout(io.StringIO()):
                export.apply_spot_check(output, "human")
        self.assertEqual(export.read_json(output / "summary.json")["exported_count"], 2)


if __name__ == "__main__":
    unittest.main()
