"""Independently verify OOD predictions and compare paired model outcomes."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from customer_policy_verifier.training import ood_choice as ood
from customer_policy_verifier.training.backbone_download import sha256
from customer_policy_verifier.training.train_jev import DEFAULT_BACKBONE, save_json, tokenizer_for


def equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            equal(a, e)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10)
    else:
        assert actual == expected


def choice(prediction):
    return max(range(len(prediction["probabilities"])), key=prediction["probabilities"].__getitem__)


def paired(before, after):
    counts = dict(both_correct=0, both_wrong=0, corrected=0, introduced_errors=0)
    assert [p["id"] for p in before] == [p["id"] for p in after]
    for old, new in zip(before, after):
        old_correct = choice(old) == old["label_id"]
        new_correct = choice(new) == new["label_id"]
        key = ("both_correct" if old_correct else "corrected") if new_correct else (
            "introduced_errors" if old_correct else "both_wrong")
        counts[key] += 1
    discordant = counts["corrected"] + counts["introduced_errors"]
    tail = min(counts["corrected"], counts["introduced_errors"])
    counts["mcnemar_exact_two_sided_p"] = min(1.0, 2 * sum(math.comb(discordant, k)
        for k in range(tail + 1)) / (2 ** discordant)) if discordant else 1.0
    counts["accuracy_delta"] = (counts["corrected"] - counts["introduced_errors"]) / len(before)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "runs/evaluation/ood-choice-v1")
    parser.add_argument("--suite", type=Path, default=ROOT / "data/evaluation/ood-choice-v1")
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/ood-choice-v1.json").read_text())
    tokenizer = tokenizer_for(args, cfg)
    manifest, files = ood.load_suite(args.suite, tokenizer)
    execution = json.loads((args.results / "execution.json").read_text())
    digest = sha256(args.suite / "manifest.json")
    assert execution["suite_manifest_sha256"] == digest
    verified, reports = {"suite_manifest_sha256": digest, "prediction_verification": "passed",
                         "models": {}, "paired_comparisons": {}}, {}
    for model in ("base", "original", "mixed"):
        verified["models"][model] = {}
        for task, n in (("ruletaker", 2), ("logiqa2", 4)):
            report = json.loads((args.results / f"{model}-{task}.json").read_text())
            assert report["model"] == model and report["task"] == task
            assert report["weight_sha256"] == execution["model_plan"][model]["sha256"]
            assert report["suite_manifest_sha256"] == digest
            for kind in ("primary", "permuted"):
                raw, encoded = files[f"{task}-{kind}.jsonl"]
                predictions = report["predictions"][kind]
                assert len(predictions) == len(raw) == len({r["id"] for r in predictions})
                for record, encoding, prediction in zip(raw, encoded, predictions):
                    assert prediction["id"] == record["id"]
                    assert prediction["label_id"] == record["label_id"]
                    assert prediction["prompt_sha256"] == encoding["prompt_sha256"]
                    for key in ("base_id", "order"):
                        if key in record:
                            assert prediction[key] == record[key]
            measured = ood.metrics(report["predictions"]["primary"], n)
            equal(report["metrics"], measured)
            diagnostic = ood.permutation_summary(report["predictions"]["primary"],
                                                 report["predictions"]["permuted"], n)
            equal(report["permutation_diagnostic"], diagnostic)
            selected_ids = {p["base_id"] for p in report["predictions"]["permuted"]}
            diagnostic["canonical_subset_metrics"] = ood.metrics(
                [p for p in report["predictions"]["primary"] if p["id"] in selected_ids], n)
            measured["ab_prediction_fraction"] = sum(measured["predicted_letter_counts"][k]
                                                       for k in "AB") / measured["samples"]
            result = {"metrics": measured, "permutation_diagnostic": diagnostic}
            if task == "logiqa2":
                seen, deduplicated = set(), []
                for raw, prediction in zip(files["logiqa2-primary.jsonl"][0], report["predictions"]["primary"]):
                    key = json.dumps([raw["context"].lower(), raw["query"].lower(),
                                      *[s.lower() for s in raw["options"]]], ensure_ascii=False)
                    if key not in seen:
                        seen.add(key)
                        deduplicated.append(prediction)
                result["deduplicated_input_metrics"] = ood.metrics(deduplicated, n)
            verified["models"][model][task] = result
            reports[model, task] = report
    for before, after in (("base", "original"), ("base", "mixed"), ("original", "mixed")):
        verified["paired_comparisons"][f"{before}_to_{after}"] = {
            task: paired(reports[before, task]["predictions"]["primary"],
                         reports[after, task]["predictions"]["primary"])
            for task in ("ruletaker", "logiqa2")}
    save_json(args.results / "verified-analysis.json", verified)
    print(json.dumps({"models": {model: {task: {"accuracy": result["metrics"]["accuracy"],
               "correct": result["metrics"]["correct"],
               "ab_fraction": result["metrics"]["ab_prediction_fraction"],
               "semantic_consistency": result["permutation_diagnostic"]["semantic_consistency"]}
               for task, result in tasks.items()} for model, tasks in verified["models"].items()},
               "paired_comparisons": verified["paired_comparisons"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
