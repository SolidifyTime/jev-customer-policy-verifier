"""Evaluate the unchanged Qwen base model using the frozen Task 2 protocol.

Uses the same encoder, native Direct model, threshold, batch size and metrics
as the completed one-epoch comparison. It does not train or select a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from customer_policy_verifier.training import jev_data
from customer_policy_verifier.training.backbone_download import sha256, source_digest
from customer_policy_verifier.training.train_jev import (
    DEFAULT_BACKBONE, DEFAULT_DATA, build_model, config_for, evaluate_rows,
    input_audit, prompt_protocol, save_json, tokenizer_for,
)

EXPECTED_DATASET = "1e0c6edd9da709a5332e1e733bfb984e24cf3bd51d897ca3de61abf9c6c3e1a4"
EXPECTED_BASE = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def numeric_equal(left, right):
    if isinstance(right, dict):
        return set(left) == set(right) and all(numeric_equal(left[k], right[k]) for k in right)
    if isinstance(right, list):
        return len(left) == len(right) and all(numeric_equal(a, b) for a, b in zip(left, right))
    if isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)
    return left == right


def validate_reference(report, rows, protocol, manifest_hash):
    if (report["dataset_manifest_sha256"] != manifest_hash or report["input_protocol"] != protocol
            or report["tiny_model"] or report["input_audit"] != input_audit(rows)):
        raise ValueError("Reference dataset, normalization, prompt or token lengths differ")
    predictions = report["predictions"]
    if len(predictions) != len(rows):
        raise ValueError("Reference sample count differs")
    for row, prediction in zip(rows, predictions):
        if (row["id"] != prediction["id"] or row["target"].index(1.0) != prediction["label_id"]
                or row["case_id"] != prediction["case_id"] or row["workflow"] != prediction["workflow"]):
            raise ValueError("Reference IDs, metadata or gold labels differ")
    recomputed = jev_data.binary_metrics([p["label_id"] for p in predictions],
                                        [p["probabilities"] for p in predictions], 0.5)
    if not numeric_equal(recomputed, report["metrics"]):
        raise ValueError("Reference probabilities do not reproduce its reported metrics")


def evaluate(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a fresh output directory")
    cfg = config_for(args)
    if (cfg.get("input_normalization") != "lowercase" or cfg["violating_threshold"] != 0.5
            or cfg["eval_batch_size"] != 1 or cfg["max_tokens"] != 2048):
        raise ValueError("Comparison requires lowercase, threshold 0.5, batch size 1 and max_tokens 2048")
    manifest_hash = sha256(args.data_dir / "manifest.json")
    if manifest_hash != EXPECTED_DATASET:
        raise ValueError("Use the previously frozen Task 2 data")
    bundle = jev_data.load_dataset_bundle(args.data_dir)
    normalization_audit = jev_data.normalization_audit(bundle, "lowercase")
    tokenizer = tokenizer_for(args, cfg)
    protocol = prompt_protocol(tokenizer, cfg)
    encoded = {split: [jev_data.encode_record(r, tokenizer, cfg["max_tokens"], "lowercase")
                       for r in bundle[split]] for split in ("validation", "test")}
    references, reference_digests = {}, {}
    expected_models = json.loads((ROOT / "configs/ood-models-autodl-v1.json").read_text())
    for name, directory in (("original", args.original_run), ("mixed", args.mixed_run)):
        selection = json.loads((directory / "selection.json").read_text())
        if selection["checkpoint_sha256"] != expected_models[name]["sha256"] or selection["threshold"] != 0.5:
            raise ValueError("Reference selected model or threshold differs")
        references[name] = {}
        for split in ("validation", "test"):
            path = directory / f"{split}-report.json"
            report = json.loads(path.read_text())
            if report["split"] != split or report["checkpoint_sha256"] != selection["checkpoint_sha256"]:
                raise ValueError("Reference split or checkpoint differs")
            validate_reference(report, encoded[split], protocol, manifest_hash)
            references[name][split] = report
            reference_digests[f"{name}/{split}"] = {"path": str(path), "sha256": sha256(path)}
    weights = args.backbone / "model.safetensors"
    if sha256(weights) != EXPECTED_BASE:
        raise ValueError("Base weights differ from the original Qwen model used in OOD evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg["gradient_checkpointing"] = False
    execution = {"started_at": now(), "model": "unfinetuned Qwen3-0.6B", "training": "none",
        "backbone_weights_sha256": EXPECTED_BASE, "backbone_source_sha256": source_digest(args.backbone),
        "dataset_manifest_sha256": manifest_hash, "config": cfg, "input_protocol": protocol,
        "normalization_audit": normalization_audit, "references": reference_digests,
        "input_audit": {split: input_audit(rows) for split, rows in encoded.items()},
        "prompt_sequence_sha256": {split: hashlib.sha256("".join(r["prompt_sha256"] for r in rows).encode()).hexdigest()
                                    for split, rows in encoded.items()},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
            [Path(__file__), *[ROOT / "src/customer_policy_verifier/training" / name for name in
              ("jev_data.py", "jev_model.py", "train_jev.py", "backbone_download.py")]]}}
    import torch
    execution["runtime"] = {"python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if cfg["device"] == "cuda" else None}
    save_json(args.output_dir / "execution.json", execution)
    status = {"state": "running", "stage": "loading_base", "started_at": now()}
    save_json(args.output_dir / "status.json", status)
    try:
        started = time.monotonic()
        model = build_model(args, cfg)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        comparison = {name: {split: report["metrics"] for split, report in splits.items()}
                      for name, splits in references.items()}
        comparison["base"] = {}
        for split, rows in encoded.items():
            status.update(stage="evaluating_" + split, samples=len(rows), updated_at=now())
            save_json(args.output_dir / "status.json", status)
            metrics, predictions = evaluate_rows(model, rows, pad, cfg)
            for row, prediction in zip(rows, predictions):
                prediction["prompt_sha256"] = row["prompt_sha256"]
            report = {"split": split, "model": "base Qwen3-0.6B", "training": "none", "tiny_model": False,
                "backbone_weights_sha256": EXPECTED_BASE, "dataset_manifest_sha256": manifest_hash,
                "input_protocol": protocol, "input_audit": input_audit(rows), "threshold": 0.5,
                "calibration": "none", "scope": "TOD-ProcBench Task 2 project-specific group split",
                "metrics": metrics, "predictions": predictions}
            save_json(args.output_dir / f"{split}-report.json", report)
            comparison["base"][split] = metrics
            print(json.dumps({"split": split, "metrics": metrics}), flush=True)
        save_json(args.output_dir / "comparison.json", comparison)
        status.update(state="complete", stage="complete", completed_at=now(), seconds=time.monotonic() - started)
        save_json(args.output_dir / "status.json", status)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}", updated_at=now())
        save_json(args.output_dir / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/jev-training-case-neutral-1epoch.json")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/evaluation/base-task2-case-neutral-v1")
    parser.add_argument("--original-run", type=Path,
                        default=Path("/root/cpv-experiments/llamafactory-direct-original-case-neutral-1epoch-v1"))
    parser.add_argument("--mixed-run", type=Path,
                        default=ROOT / "runs/training/llamafactory-direct-mixed-case-neutral-1epoch-v1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.set_defaults(tiny_model=False)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
