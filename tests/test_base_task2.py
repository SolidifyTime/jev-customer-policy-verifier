"""Ensure the base/fine-tuned comparison cannot silently change its protocol."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from customer_policy_verifier.training import jev_data
from customer_policy_verifier.training.train_jev import input_audit

spec = importlib.util.spec_from_file_location("evaluate_base_task2", PROJECT / "scripts/evaluate_base_task2.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class BaseTask2ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": "frozen", "input_ids": [1, 2, 3], "target": [0.0, 1.0],
                      "case_id": "42", "workflow": "refund"}]
        self.protocol = {"input_normalization": "lowercase", "option_letters": ["A", "B"]}
        predictions = [{"id": "frozen", "label_id": 1, "case_id": "42", "workflow": "refund",
                        "probabilities": [0.2, 0.8]}]
        self.report = {"dataset_manifest_sha256": "fixed", "input_protocol": self.protocol,
                       "tiny_model": False, "input_audit": input_audit(self.rows), "predictions": predictions,
                       "metrics": jev_data.binary_metrics([1], [[0.2, 0.8]])}

    def test_reference_protocol_and_probabilities_are_verified(self):
        base.validate_reference(self.report, self.rows, self.protocol, "fixed")
        for key, value in (("dataset_manifest_sha256", "different"), ("tiny_model", True),
                           ("input_protocol", {"input_normalization": "none"}),
                           ("input_audit", {"truncated": 1})):
            report = {**self.report, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                base.validate_reference(report, self.rows, self.protocol, "fixed")

    def test_swapped_gold_or_changed_sample_order_cannot_be_compared(self):
        for change in ({"id": "other"}, {"label_id": 0}, {"case_id": "other"}, {"workflow": "other"}):
            report = copy.deepcopy(self.report)
            report["predictions"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                base.validate_reference(report, self.rows, self.protocol, "fixed")
        report = {**self.report, "predictions": []}
        with self.assertRaises(ValueError):
            base.validate_reference(report, self.rows, self.protocol, "fixed")

    def test_modified_reference_scores_are_rejected(self):
        report = copy.deepcopy(self.report)
        report["metrics"]["accuracy"] = 0.0
        with self.assertRaisesRegex(ValueError, "probabilities"):
            base.validate_reference(report, self.rows, self.protocol, "fixed")


if __name__ == "__main__":
    unittest.main()
