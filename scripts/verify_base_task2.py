"""Verify Task 2 base predictions on a machine with tokenizer files only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import evaluate_base_task2 as base
from customer_policy_verifier.training import jev_data
from customer_policy_verifier.training.train_jev import DEFAULT_BACKBONE, DEFAULT_DATA, save_json, tokenizer_for, prompt_protocol
from customer_policy_verifier.training.backbone_download import sha256

ROOT = base.ROOT


def correct(prediction):
    return int(prediction["probabilities"][1] >= 0.5) == prediction["label_id"]


def paired(before, after):
    counts = {"both_correct": 0, "both_wrong": 0, "corrected": 0, "introduced_errors": 0}
    if [p["id"] for p in before] != [p["id"] for p in after]:
        raise ValueError("Paired comparison IDs differ")
    for old, new in zip(before, after):
        old_correct, new_correct = correct(old), correct(new)
        key = ("both_correct" if old_correct else "corrected") if new_correct else (
            "introduced_errors" if old_correct else "both_wrong")
        counts[key] += 1
    counts["accuracy_gain_percentage_points"] = 100 * (counts["corrected"] - counts["introduced_errors"]) / len(before)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "runs/evaluation/base-task2-case-neutral-v1")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    reference_root = ROOT / "runs/maintenance/case-neutral-1epoch-20261004/remote-results"
    parser.add_argument("--original-run", type=Path, default=reference_root / "original")
    parser.add_argument("--mixed-run", type=Path, default=reference_root / "mixed")
    args = parser.parse_args()
    execution = json.loads((args.results / "execution.json").read_text())
    if (execution["training"] != "none" or execution["backbone_weights_sha256"] != base.EXPECTED_BASE
            or sha256(args.data_dir / "manifest.json") != base.EXPECTED_DATASET
            or execution["dataset_manifest_sha256"] != base.EXPECTED_DATASET):
        raise ValueError("Base identity, dataset or no-training record differs")
    for path, digest in execution["source_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("Frozen evaluation code differs")
    cfg = execution["config"]
    if (cfg["input_normalization"] != "lowercase" or cfg["violating_threshold"] != 0.5
            or cfg["eval_batch_size"] != 1 or cfg["max_tokens"] != 2048):
        raise ValueError("Frozen evaluation config differs")
    tokenizer = tokenizer_for(SimpleNamespace(backbone=args.backbone), cfg)
    protocol = prompt_protocol(tokenizer, cfg)
    if protocol != execution["input_protocol"]:
        raise ValueError("Tokenizer/prompt protocol differs")
    bundle = jev_data.load_dataset_bundle(args.data_dir)
    normalization_audit = jev_data.normalization_audit(bundle, "lowercase")
    if normalization_audit != execution["normalization_audit"]:
        raise ValueError("Normalized dataset audit differs")
    comparison = json.loads((args.results / "comparison.json").read_text())
    verified = {"state": "complete", "predictions_verified": 0, "models": {}, "paired": {}}
    reports = {}
    for split in ("validation", "test"):
        encoded = [jev_data.encode_record(row, tokenizer, cfg["max_tokens"], "lowercase") for row in bundle[split]]
        for name, directory in (("base", args.results), ("original", args.original_run), ("mixed", args.mixed_run)):
            path = directory / f"{split}-report.json"
            report = json.loads(path.read_text())
            if report["split"] != split:
                raise ValueError("Report split differs")
            base.validate_reference(report, encoded, protocol, base.EXPECTED_DATASET)
            if name == "base":
                if report["training"] != "none" or report["threshold"] != 0.5 or report["backbone_weights_sha256"] != base.EXPECTED_BASE:
                    raise ValueError("Base report identity differs")
                for prediction, encoding in zip(report["predictions"], encoded):
                    if prediction["prompt_sha256"] != encoding["prompt_sha256"]:
                        raise ValueError("Base rendered prompt hash differs")
            elif sha256(path) != execution["references"][f"{name}/{split}"]["sha256"]:
                raise ValueError("Reference report changed")
            measured = jev_data.binary_metrics([p["label_id"] for p in report["predictions"]],
                                               [p["probabilities"] for p in report["predictions"]], 0.5)
            if not base.numeric_equal(measured, comparison[name][split]):
                raise ValueError("Comparison score differs from recomputed probabilities")
            verified["models"].setdefault(name, {})[split] = measured
            verified["predictions_verified"] += len(report["predictions"])
            reports[name, split] = report
        verified["paired"][split] = {name: paired(reports["base", split]["predictions"],
                                                  reports[name, split]["predictions"])
                                     for name in ("original", "mixed")}
    save_json(args.results / "verified-analysis.json", verified)
    print(json.dumps(verified, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
