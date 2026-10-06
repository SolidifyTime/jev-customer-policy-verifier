"""Synthetic, offline regression checks for source coverage and split isolation."""

import collections
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from customer_policy_verifier.common.paths import resolve_path
from customer_policy_verifier.data import prepare_full_data as full


class FullDataTests(unittest.TestCase):
    def setUp(self):
        # Public source-only checkouts contain no benchmark files. Build nine
        # pinned views and a historical split entirely from synthetic text.
        temporary = tempfile.TemporaryDirectory(prefix="cpv-source-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "task2"
        raw_dir = self.data / "raw"
        raw_dir.mkdir(parents=True)
        assignments = full.build_split([{"orig_conv_id": group} for group in range(498)], "0" * 64)["group_assignment"]
        self.rows = []
        # Match the original regression's scale and split counts without
        # depending on any real source text, conversation ids or source hash.
        for partition, base_pairs, extra_groups in (("train", 4, 26), ("validation", 3, 65), ("test", 3, 98)):
            groups = sorted(int(group) for group, assigned in assignments.items() if assigned == partition)
            for position, group in enumerate(groups):
                for _ in range(base_pairs + int(position < extra_groups)):
                    index = len(self.rows)
                    self.rows.append({
                        "orig_conv_id": group,
                        "intent": f"fixture_intent_{index % 55}",
                        "instruction": f"Synthetic policy {index}: ask for the fixture order id before checking.",
                        "conversation": [f"turn 0: customer: Please check synthetic order {index}."],
                        "orig_gt_response": f"turn 1: agent: Please provide fixture order id {index}.",
                        "manipulated_gt_response": (
                            f"Unprefixed synthetic negative reply {index}." if index < 45 else
                            f"turn 1: agent: Synthetic order {index} is complete without checking."
                        ),
                    })

        catalog = []
        for name, _, _ in full.FILES:
            if name == "test_task2_en_f1.json":
                view = self.rows
            elif name in ("test_task2_en_f2.json", "test_task2_en_f3.json"):
                ordered = list(reversed(self.rows)) if "_f2" in name else self.rows[1:] + self.rows[:1]
                view = [{**row, "instruction": "Alternate fixture formatting: " + row["instruction"]} for row in ordered]
            else:
                language = name.split("_")[-2]
                view = [{**row, "conversation": [f"turn 0: customer: {language} fixture case {index}."],
                         "orig_gt_response": f"turn 1: agent: {language} fixture positive {index}.",
                         "manipulated_gt_response": f"turn 1: agent: {language} fixture negative {index}."}
                        for index, row in enumerate(self.rows)]
            path = raw_dir / name
            path.write_bytes(full.serialized(view))
            catalog.append((name, "fixture-" + name, hashlib.sha256(path.read_bytes()).hexdigest()))

        metadata_catalog = []
        for name, _, _ in full.METADATA_FILES:
            payload = f"Synthetic offline fixture metadata: {name}.\n".encode()
            (self.data / name).write_bytes(payload)
            metadata_catalog.append((name, "fixture-" + name, hashlib.sha256(payload).hexdigest()))
        raw = raw_dir / "test_task2_en_f1.json"
        split = self.root / "historical-split.json"
        split.write_bytes(full.serialized(full.build_split(self.rows, hashlib.sha256(raw.read_bytes()).hexdigest())))
        exclusions = self.root / "fixture-exclusions.jsonl"
        exclusions.write_text("".join(json.dumps(row) + "\n" for row in (
            {"source_row_index": 1028, "expected_label": "violating", "label_supported": "否", "reason": "Fixture negative exclusion."},
            {"source_row_index": 1177, "expected_label": "compliant", "label_supported": "无法确定", "reason": "Fixture uncertain exclusion."},
        )), encoding="utf-8")
        changes = {"FILES": tuple(catalog), "METADATA_FILES": tuple(metadata_catalog),
                   "RAW": raw, "SPLIT": split, "EXCLUSIONS": exclusions}
        for name, value in changes.items():
            replacement = patch.object(full, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        for target in ("urllib.request.urlopen", "subprocess.run"):
            no_network = patch(target, side_effect=AssertionError("Fixture preparation must not use network"))
            no_network.start()
            self.addCleanup(no_network.stop)

    def test_all_views_and_no_split_leakage(self):
        manifest = full.prepare(data=self.data)
        self.assertEqual(manifest["scope"]["official_task2_json_files"], 9)
        self.assertEqual(manifest["scope"]["unique_source_pairs"], 1982)
        self.assertEqual(manifest["source_sha256"], full.sha256(full.RAW))
        self.assertEqual(full.sha256(resolve_path(manifest["split_file"])), full.sha256(full.SPLIT))
        group_sets = []
        pairs = collections.defaultdict(list)
        for partition, path in manifest["originals"].items():
            records = [json.loads(line) for line in resolve_path(path).read_text().splitlines()]
            self.assertEqual(len(records), {"train": 2444, "validation": 730, "test": 790}[partition])
            group_sets.append({item["metadata"]["orig_conv_id"] for item in records})
            for item in records:
                self.assertEqual(item["metadata"]["split"], partition)
                pairs[item["metadata"]["source_row_index"]].append((partition, item["label"]))
        self.assertEqual(len(pairs), 1982)
        self.assertFalse(group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
        for records in pairs.values():
            self.assertEqual(len({part for part, _ in records}), 1)
            self.assertEqual({label for _, label in records}, {"compliant", "violating"})
        eligibility = manifest["generation_eligibility"]
        self.assertEqual(eligibility["eligible_candidates"], 2397)
        self.assertEqual(eligibility["label_counts"], {"compliant": 1221, "violating": 1176})

    def test_repeated_preparation_is_byte_identical(self):
        full.prepare(data=self.data)
        manifest_path = self.data / "manifest.json"
        before = manifest_path.read_bytes()
        full.prepare(data=self.data)
        self.assertEqual(before, manifest_path.read_bytes())

    def test_reordered_format_view_is_not_additional_data(self):
        rows = full.load_rows(full.RAW)[:3]
        alternate = copy.deepcopy(list(reversed(rows)))
        for row in alternate:
            row["instruction"] = "Alternate formatting"
        audit = full.audit_views({"test_task2_en_f1.json": rows, "test_task2_en_f2.json": alternate})
        self.assertEqual(audit["test_task2_en_f2.json"]["new_base_pairs"], 0)
        alternate[0]["candidate_extra"] = "Different source"
        with self.assertRaisesRegex(ValueError, "different source samples"):
            full.audit_views({"test_task2_en_f1.json": rows, "test_task2_en_f2.json": alternate})

    def test_translation_alignment_change_is_rejected(self):
        rows = full.load_rows(full.RAW)[:3]
        translated = copy.deepcopy(rows)
        translated[0]["orig_conv_id"] = -1
        with self.assertRaisesRegex(ValueError, "does not align"):
            full.audit_views({"test_task2_en_f1.json": rows, "test_task2_zh_f1.json": translated})

    def test_pinned_file_tampering_and_frozen_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "file.json"
            full.freeze(path, b"original")
            full.freeze(path, b"original")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                full.freeze(path, b"changed")
            with self.assertRaisesRegex(ValueError, "differs from pinned"):
                full.obtain(path, "unused", hashlib.sha256(b"changed").hexdigest(), False)

    def test_changed_historical_group_assignment_is_rejected(self):
        rows = full.load_rows(full.RAW)
        frozen = json.loads(full.SPLIT.read_bytes())
        group = next(iter(frozen["group_assignment"]))
        frozen["group_assignment"][group] = "test" if frozen["group_assignment"][group] == "train" else "train"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.json"
            path.write_text(json.dumps(frozen))
            with patch.object(full, "SPLIT", path), self.assertRaisesRegex(ValueError, "historical frozen split"):
                full.split_data(rows, full.sha256(full.RAW))


if __name__ == "__main__":
    unittest.main()
