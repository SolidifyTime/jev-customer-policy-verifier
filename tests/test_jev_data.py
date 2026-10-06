"""Offline checks for the fixed bundle, input isolation and classification metrics."""

import collections
import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from customer_policy_verifier.training import jev_data as data


SOURCE_DIGEST = "a" * 64


def record(identifier="original-secret-label", group=1, split="train", label="compliant", source_type="original"):
    return {"id": identifier, "policy": "Ask for order details before checking shipping.",
            "conversation": ["turn 0: customer: When will my order arrive?"],
            "candidate_response": "turn 1: agent: Please provide your order ID.",
            "label": label,
            "metadata": {"orig_conv_id": group, "intent": "status_shipping", "split": split,
                         "source_type": source_type, "source_sha256": SOURCE_DIGEST,
                         "review_status": "secret-review-verdict"}}


def fixture_records():
    good = record("train-good")
    bad = record("train-bad", label="violating")
    bad["candidate_response"] = "turn 1: agent: Your order has shipped, no details needed."
    synthetic = record("synthetic", source_type="synthetic")
    synthetic["conversation"] = ["turn 0: customer: Could you check my shipping date?"]
    validation = record("validation", 2, "validation")
    validation["conversation"] = ["turn 0: customer: Please check order two."]
    test = record("test", 3, "test", "violating")
    test["conversation"] = ["turn 0: customer: Please check order three."]
    return {"train": [synthetic, bad, good], "train_original": [good, bad],
            "validation": [validation], "test": [test]}


def write_bundle(directory, records):
    manifest = {"version": "training-data-v1", "seed": 20261002,
                "model_input_fields": list(data.MODEL_FIELDS), "target_field": "label",
                "label_to_id": {"compliant": 0, "violating": 1},
                "source_sha256": SOURCE_DIGEST, "split_sha256": "b" * 64,
                "inputs": [], "files": {}}
    for key, filename in data.FILE_NAMES.items():
        rows = records[key]
        payload = ("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)).encode("utf-8")
        (directory / filename).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        manifest["files"][filename] = {"count": len(rows), "sha256": digest,
                                       "label_counts": dict(collections.Counter(r["label"] for r in rows))}
        if key in ("validation", "test"):
            # The archive is deliberately absent, as on a different GPU machine.
            manifest["inputs"].append({"path": f"/other-machine/data/task2/originals/{filename}", "sha256": digest})
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class FakeTokenizer:
    def __init__(self):
        self.calls = []
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((copy.deepcopy(messages), kwargs))
        return json.dumps(messages, ensure_ascii=False) + "\n<assistant>"

    def encode(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        if text in data.OPTION_LETTERS:
            return [{"A": 32, "B": 33}[text]]
        if text.endswith("\n<assistant>A") or text.endswith("\n<assistant>B"):
            return [1000] + list(text[:-1].encode("utf-8")) + [{"A": 32, "B": 33}[text[-1]]]
        return [1000] + list(text.encode("utf-8"))

    def decode(self, ids):
        return {(32,): "A", (33,): "B"}.get(tuple(ids), "?")


class JevDataTests(unittest.TestCase):
    def test_lowercase_evidence_is_invariant_to_case_without_mutating_records(self):
        item = record()
        before = copy.deepcopy(item)
        changed = copy.deepcopy(item)
        for key in data.MODEL_FIELDS:
            changed[key] = ([turn.swapcase() for turn in item[key]] if isinstance(item[key], list)
                            else item[key].swapcase())
        left = data.encode_record(item, FakeTokenizer(), 10000, "lowercase")
        right = data.encode_record(changed, FakeTokenizer(), 10000, "lowercase")
        self.assertEqual(left["input_ids"], right["input_ids"])
        self.assertEqual(left["prompt_sha256"], right["prompt_sha256"])
        self.assertEqual(item, before)
        self.assertEqual(left["target"], [1.0, 0.0])
        evidence = json.loads(left["state"])
        self.assertTrue(all(value == value.lower() for key in data.MODEL_FIELDS
                            for value in (evidence[key] if isinstance(evidence[key], list) else [evidence[key]])))
        payload = json.loads(data.direct_messages(item, "lowercase")[1]["content"])
        self.assertEqual([option["letter"] for option in payload["options"]], ["A", "B"])
        self.assertEqual(data.direct_messages(item, "lowercase"),
                         data.direct_messages({key: item[key] for key in data.MODEL_FIELDS}, "lowercase"))

    def test_normalization_rejects_case_only_label_conflicts_and_cross_split_overlap(self):
        bundle = fixture_records()
        duplicate = copy.deepcopy(bundle["train"][1])
        duplicate["candidate_response"] = bundle["train"][2]["candidate_response"].upper()
        bundle["train"][1] = duplicate
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            data.normalization_audit(bundle, "lowercase")
        bundle = fixture_records()
        duplicate = copy.deepcopy(bundle["validation"][0])
        for key in data.MODEL_FIELDS:
            value = bundle["train"][0][key]
            duplicate[key] = ([turn.upper() for turn in value] if isinstance(value, list) else value.upper())
        bundle["validation"][0] = duplicate
        with self.assertRaisesRegex(ValueError, "cross partitions"):
            data.normalization_audit(bundle, "lowercase")

    def test_normalization_audit_keeps_frozen_counts_and_rejects_unknown_transform(self):
        bundle = fixture_records()
        audit = data.normalization_audit(bundle, "lowercase")
        self.assertEqual(audit["integrity"], "passed")
        self.assertEqual({key: spec["count"] for key, spec in audit["splits"].items()},
                         {key: len(bundle[key]) for key in data.FILE_NAMES})
        for spec in audit["splits"].values():
            self.assertEqual(spec["responses_with_uppercase_by_label"], {"compliant": 0, "violating": 0})
        with self.assertRaisesRegex(ValueError, "Unknown input_normalization"):
            data.state_text(record(), "unexpected")

    def test_state_excludes_answers_and_management_fields(self):
        item = record()
        item["generation_note"] = "secret-generation-note"
        item["metadata"]["source_row_index"] = 12345
        text = data.state_text(item)
        self.assertEqual(json.loads(text), {k: item[k] for k in data.MODEL_FIELDS})
        for secret in (item["id"], "secret-review-verdict", "secret-generation-note", "12345", '"label"', '"metadata"'):
            self.assertNotIn(secret, text)
        reordered = dict(reversed(list(item.items())))
        self.assertEqual(text, data.state_text(reordered))
        decision = data.to_question(item)
        self.assertEqual(decision["target"], [1.0, 0.0])
        self.assertEqual(decision["question"]["keys"], list(data.LABELS))
        self.assertEqual(decision["case_id"], "1")
        item["label"] = "violating"
        self.assertEqual(data.to_question(item)["target"], [0.0, 1.0])
        self.assertEqual(text, data.state_text(item))

    def test_encoding_matches_semif_direct_reference_format(self):
        item, tokenizer = record(), FakeTokenizer()
        encoded = data.encode_record(item, tokenizer, 10000)
        # The message/payload structure and template flags are SemIf Direct's
        # public contract, independently of the dataset's management fields.
        expected_system = (
            "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
            "Respond with only its uppercase letter, with no explanation or reasoning."
        )
        expected_payload = {
            "evidence": data.state_text(item), "criterion": data.QUESTION,
            "options": [{"letter": "A", "description": data.CANDIDATES[0]},
                        {"letter": "B", "description": data.CANDIDATES[1]}],
        }
        expected_messages = [
            {"role": "system", "content": expected_system},
            {"role": "user", "content": json.dumps(expected_payload, ensure_ascii=False)},
        ]
        self.assertEqual(tokenizer.template_calls, [(expected_messages, {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        })])
        prompt = json.dumps(expected_messages, ensure_ascii=False) + "\n<assistant>"
        self.assertEqual(encoded["input_ids"], [1000] + list(prompt.encode("utf-8")))
        self.assertEqual(encoded["option_token_ids"], [32, 33])
        self.assertEqual(encoded["prompt_sha256"], hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        self.assertEqual(data.PROMPT_VERSION, "semif-direct-v1")
        self.assertEqual([text for text, _ in tokenizer.calls], [prompt, "A", "B", prompt + "A", prompt + "B"])
        self.assertEqual(set(encoded), {"id", "case_id", "workflow", "state", "question", "target",
                                        "input_ids", "option_token_ids", "prompt_sha256"})
        self.assertNotIn("paths", encoded)
        for text, special in tokenizer.calls:
            self.assertFalse(special)
            self.assertNotIn(item["id"], text)
            self.assertNotIn("secret-review-verdict", text)

    def test_prompt_ignores_labels_ids_and_management_metadata(self):
        item = record()
        original = data.encode_record(item, FakeTokenizer(), 10000)
        changed = copy.deepcopy(item)
        changed.update(id="another-secret-id", label="violating", generation_note="SECRET-GENERATION")
        changed["metadata"].update(orig_conv_id=987654, intent="SECRET-INTENT", review_status="SECRET-REVIEW")
        modified = data.encode_record(changed, FakeTokenizer(), 10000)
        self.assertEqual(original["input_ids"], modified["input_ids"])
        self.assertEqual(original["prompt_sha256"], modified["prompt_sha256"])
        self.assertEqual(original["target"], [1.0, 0.0])
        self.assertEqual(modified["target"], [0.0, 1.0])
        # Message construction itself supports unlabeled inference inputs.
        inference = {key: item[key] for key in data.MODEL_FIELDS}
        self.assertEqual(data.direct_messages(inference), data.direct_messages(item))
        payload = json.loads(data.direct_messages(item)[1]["content"])
        self.assertEqual(set(payload), {"evidence", "criterion", "options"})
        self.assertEqual(json.loads(payload["evidence"]), inference)

    def test_encoding_refuses_long_prompts_and_invalid_limits(self):
        item = record()
        exact_length = len(data.encode_record(item, FakeTokenizer(), 10000)["input_ids"])
        self.assertEqual(len(data.encode_record(item, FakeTokenizer(), exact_length)["input_ids"]), exact_length)
        with self.assertRaisesRegex(ValueError, "no truncation allowed"):
            data.encode_record(item, FakeTokenizer(), exact_length - 1)
        for limit in (True, 0, -1, 1.5):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "positive integer"):
                data.encode_record(item, FakeTokenizer(), limit)

    def test_answer_slot_requires_one_round_trip_token(self):
        tokenizer = FakeTokenizer()
        original_encode = tokenizer.encode
        tokenizer.encode = lambda text, **kwargs: [32, 99] if text == "A" else original_encode(text, **kwargs)
        with self.assertRaisesRegex(ValueError, "round-trip token"):
            data.slot_token_ids(tokenizer)
        tokenizer = FakeTokenizer()
        tokenizer.decode = lambda ids: "wrong letter"
        with self.assertRaisesRegex(ValueError, "round-trip token"):
            data.slot_token_ids(tokenizer)

    def test_answer_slots_cannot_share_token_id(self):
        class CollidingTokenizer(FakeTokenizer):
            def encode(self, text, **kwargs):
                self.letter = text
                return [32]

            def decode(self, ids):
                return self.letter

        with self.assertRaisesRegex(ValueError, "distinct"):
            data.slot_token_ids(CollidingTokenizer())

    def test_prompt_answer_boundary_must_preserve_prefix_tokenization(self):
        class MergingTokenizer(FakeTokenizer):
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                return [999] + ids[1:] if text.endswith("\n<assistant>A") else ids

        with self.assertRaisesRegex(ValueError, "boundary changes tokenization"):
            data.encode_record(record(), MergingTokenizer(), 10000)

    def test_empty_or_invalid_template_and_token_ids_are_rejected(self):
        for prompt in ("", None, [1, 2]):
            tokenizer = FakeTokenizer()
            tokenizer.apply_chat_template = lambda *args, **kwargs: prompt
            with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, "nonempty string"):
                data.encode_record(record(), tokenizer, 10000)
        for ids in ([], [True], [-1], [1.5], (1, 2)):
            tokenizer = FakeTokenizer()
            tokenizer.encode = lambda *args, **kwargs: ids
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "integer token ids"):
                data.encode_record(record(), tokenizer, 10000)

    def test_same_history_positive_and_negative_responses_remain_distinct(self):
        rows = fixture_records()
        good, bad = rows["train_original"]
        self.assertEqual(good["conversation"], bad["conversation"])
        self.assertNotEqual(data.state_text(good), data.state_text(bad))
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_bundle(directory, rows)
            bundle = data.load_dataset_bundle(directory)
            self.assertEqual(bundle["train"], rows["train"])
            self.assertEqual(bundle["train_original"], rows["train_original"])
            self.assertEqual(bundle["validation"], rows["validation"])
            self.assertEqual(bundle["test"], rows["test"])

    def test_tampered_file_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_bundle(directory, fixture_records())
            with (directory / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                data.load_dataset_bundle(directory)

    def test_changed_manifest_counts_or_mapping_are_rejected(self):
        for mutation, message in ((lambda m: m["files"]["train.jsonl"].update(count=999), "count mismatch"),
                                  (lambda m: m.update(label_to_id={"compliant": 1, "violating": 0}), "Label mapping"),
                                  (lambda m: m["files"]["train.jsonl"].update(label_counts={"compliant": 3}), "Label counts")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                manifest = write_bundle(directory, fixture_records())
                mutation(manifest)
                (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    data.load_dataset_bundle(directory)

    def test_original_subset_mismatch_is_rejected_after_hash_refresh(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            rows = fixture_records()
            rows["train_original"] = copy.deepcopy(rows["train_original"])
            rows["train_original"][0]["candidate_response"] += " Changed."
            write_bundle(directory, rows)
            with self.assertRaisesRegex(ValueError, "original subset"):
                data.load_dataset_bundle(directory)

    def test_frozen_validation_digest_cannot_be_replaced_by_output_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            rows = fixture_records()
            original_manifest = write_bundle(directory, rows)
            rows["validation"][0]["candidate_response"] += " Changed."
            refreshed = write_bundle(directory, rows)
            refreshed["inputs"] = original_manifest["inputs"]
            (directory / "manifest.json").write_text(json.dumps(refreshed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen original validation"):
                data.load_dataset_bundle(directory)

    def test_group_leakage_is_rejected_even_with_matching_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            rows = fixture_records()
            rows["test"][0]["metadata"]["orig_conv_id"] = 1
            write_bundle(directory, rows)
            with self.assertRaisesRegex(ValueError, "groups cross partitions"):
                data.load_dataset_bundle(directory)

    def test_identical_complete_inputs_across_groups_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            rows = fixture_records()
            for key in data.MODEL_FIELDS:
                rows["test"][0][key] = copy.deepcopy(rows["train_original"][0][key])
            write_bundle(directory, rows)
            with self.assertRaisesRegex(ValueError, "Complete inputs cross partitions"):
                data.load_dataset_bundle(directory)

    def test_metrics_include_each_class_and_probability_quality(self):
        targets = [0, 0, 0, 1, 1]
        probs = [[0.8, 0.2], [0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]]
        metrics = data.binary_metrics(targets, probs)
        self.assertEqual(metrics["n"], 5)
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [1, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["macro_f1"], ((4 / 6) + (2 / 4)) / 2)
        self.assertAlmostEqual(metrics["violating_recall"], 0.5)
        self.assertAlmostEqual(metrics["compliant_false_positive_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["cross_entropy"], -sum(math.log(p[y]) for y, p in zip(targets, probs)) / 5)
        expected_brier = sum(sum((p - float(k == y)) ** 2 for k, p in enumerate(v)) for y, v in zip(targets, probs)) / 5
        self.assertAlmostEqual(metrics["brier_sum"], expected_brier)
        self.assertEqual(data.binary_metrics([0], [[0.5, 0.5]])["confusion_matrix"], [[0, 1], [0, 0]])
        self.assertEqual(data.binary_metrics([1], [[0.4, 0.6]], threshold=0.7)["confusion_matrix"], [[0, 0], [1, 0]])

    def test_metrics_invalid_or_nonfinite_inputs_are_rejected(self):
        invalid = [([], []), ([0], []), ([2], [[0.5, 0.5]]),
                   ([True], [[0.5, 0.5]]), ([0], [[0.5]]),
                   ([0], [[-0.1, 1.1]]), ([0], [[float("nan"), 0.5]]),
                   ([0], [[float("inf"), 0.5]]), ([0], [[0.7, 0.7]])]
        for targets, probs in invalid:
            with self.subTest(targets=targets, probs=probs), self.assertRaises(ValueError):
                data.binary_metrics(targets, probs)
        for threshold in (True, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                data.binary_metrics([0], [[1.0, 0.0]], threshold=threshold)
        self.assertTrue(math.isfinite(data.binary_metrics([1], [[1.0, 0.0]])["cross_entropy"]))


if __name__ == "__main__":
    unittest.main()
