"""Choice evaluation integrity, gold isolation and option permutation checks."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from customer_policy_verifier.training import ood_choice as ood


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return json.dumps(messages) + "\n"

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(map(chr, ids))


def rule_source():
    return [{"id": f"theory-{group}_{label}",
             "theory_assertion_instance": {"label": label == 0, "exception": None,
                                           "proof": "HIDDEN_PROOF"},
             "english": {"theory_statements": [f"Bob is blue in theory {group}."],
                         "assertion_statement": f"Own query {group}/{label}."},
             "logical_forms": "HIDDEN_FORMULA"}
            for group in range(8) for label in (0, 1)]


def choice_row():
    return {"id": "private-ID", "task": "logiqa2", "context": "Given EVIDENCE.",
            "query": "Which follows?", "options": ["First", "Second", "Third", "Fourth"],
            "label_id": 2, "proof": "private-PROOF"}


class OODChoiceTests(unittest.TestCase):
    def test_rule_sampling_is_balanced_unique_grouped_and_uses_own_label(self):
        source = rule_source()
        before = copy.deepcopy(source)
        selected = ood.rule_subset(source, count=6, seed=17)
        self.assertEqual(selected, ood.rule_subset(source, count=6, seed=17))
        self.assertEqual(len({r["group_id"] for r in selected}), 6)
        self.assertEqual(sorted(r["label_id"] for r in selected), [0, 0, 0, 1, 1, 1])
        by_id = {r["id"]: r for r in source}
        for row in selected:
            original = by_id[row["id"]]
            self.assertEqual(row["query"], original["english"]["assertion_statement"])
            self.assertEqual(row["label_id"], int(not original["theory_assertion_instance"]["label"]))
            self.assertNotIn("proof", row)
        self.assertEqual(source, before)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ood.rule_subset(source + [source[0]], count=6)

    def test_problog_exception_and_missing_labels_are_rejected(self):
        source = rule_source()
        source[0]["theory_assertion_instance"]["exception"] = "inconsistent"
        with self.assertRaises(ValueError):
            ood.rule_subset(source, count=6)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            ood.rule_subset(rule_source(), count=10)

    def test_gold_and_management_metadata_cannot_change_model_input(self):
        row = choice_row()
        changed = {**row, "id": "different-ID", "label_id": 0, "proof": "different-PROOF"}
        tokenizer = CharacterTokenizer()
        first, second = ood.encode(row, tokenizer, 4096), ood.encode(changed, tokenizer, 4096)
        self.assertEqual(first["input_ids"], second["input_ids"])
        self.assertEqual(first["prompt_sha256"], second["prompt_sha256"])
        self.assertNotEqual(first["target"], second["target"])
        prompt = json.dumps(ood.messages(row))
        self.assertNotIn("private", prompt)
        self.assertNotIn("EVIDENCE", prompt)
        self.assertEqual(tokenizer.decode(first["option_token_ids"]), "ABCD")
        upper = {**row, "context": row["context"].upper(), "query": row["query"].upper(),
                 "options": [s.upper() for s in row["options"]]}
        self.assertEqual(first["input_ids"], ood.encode(upper, tokenizer, 4096)["input_ids"])

    def test_encoding_never_truncates_and_requires_answer_boundary(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            ood.encode(choice_row(), CharacterTokenizer(), 5)
        class BrokenBoundary(CharacterTokenizer):
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                return ids[:-2] + [999] if text.endswith("\nA") else ids
        with self.assertRaisesRegex(ValueError, "boundary"):
            ood.encode(choice_row(), BrokenBoundary(), 4096)

    def test_permutation_maps_gold_and_semantic_consistency(self):
        row = choice_row()
        variant = ood.permute(row, [1, 2, 3, 0])
        self.assertEqual(variant["label_id"], 1)
        self.assertEqual(variant["options"][variant["label_id"]], row["options"][row["label_id"]])
        primary = [{"id": row["id"], "label_id": 2, "probabilities": [0.1, 0.1, 0.7, 0.1]}]
        rotated = [{"id": variant["id"], "base_id": row["id"], "order": variant["order"],
                    "label_id": 1, "probabilities": [0.1, 0.7, 0.1, 0.1]}]
        self.assertEqual(ood.permutation_summary(primary, rotated, 4)["semantic_consistency"], 1)
        rotated[0]["probabilities"] = [0.1, 0.1, 0.7, 0.1]
        self.assertEqual(ood.permutation_summary(primary, rotated, 4)["semantic_consistency"], 0)
        with self.assertRaises(ValueError):
            ood.permute(row, [0, 0, 2, 3])

    def test_four_choice_metrics_and_probability_validation(self):
        predictions = [{"id": str(i), "label_id": i, "probabilities": [0.7 if j == i else 0.1 for j in range(4)]}
                       for i in range(4)]
        measured = ood.metrics(predictions, 4)
        self.assertEqual(measured["accuracy"], 1)
        self.assertEqual(measured["macro_f1"], 1)
        self.assertEqual(measured["majority_gold_accuracy"], 0.25)
        self.assertAlmostEqual(measured["restricted_cross_entropy"], -math.log(0.7))
        predictions[0]["probabilities"] = [float("nan"), 0.1, 0.1, 0.1]
        with self.assertRaises(ValueError):
            ood.metrics(predictions, 4)

    def test_duplicate_official_logiqa_ids_preserve_distinct_source_rows(self):
        source = [{"id": 1, "answer": 2, "text": "Passage", "question": "Question",
                   "options": ["a", "b", "c", "d"]}] * 2
        rows = ood.logiqa_rows(source)
        self.assertEqual(len({r["id"] for r in rows}), 2)
        self.assertEqual([r["source_row"] for r in rows], [1, 2])
        self.assertEqual(ood.input_quality(rows), {"unique_complete_inputs": 1,
                          "duplicate_extra_rows": 1, "conflicting_label_groups": 0})

    def test_frozen_file_and_rendered_prompt_hashes_are_checked(self):
        tokenizer = CharacterTokenizer()
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)
            rows = [choice_row()]
            name = "logiqa2-primary.jsonl"
            ood.write_jsonl(suite / name, rows)
            encoded = ood.encode(rows[0], tokenizer, 4096)
            manifest = {"version": ood.VERSION, "system": ood.DIRECT_SYSTEM,
                        "criteria": {"ruletaker": ood.RULE_CRITERION, "logiqa2": ood.LOGIC_CRITERION},
                        "max_tokens": 4096, "files": {name: {"sha256": ood.sha256(suite / name), "samples": 1,
                        "prompt_sequence_sha256": hashlib.sha256(encoded["prompt_sha256"].encode()).hexdigest()}}}
            (suite / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(ood.load_suite(suite, tokenizer)[1][name][0], rows)
            manifest["files"][name]["prompt_sequence_sha256"] = "0" * 64
            (suite / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "tokenizer/prompt"):
                ood.load_suite(suite, tokenizer)
            (suite / name).write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "data hash"):
                ood.load_suite(suite, tokenizer)


if __name__ == "__main__":
    unittest.main()
