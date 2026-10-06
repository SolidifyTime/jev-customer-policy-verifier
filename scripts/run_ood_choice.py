"""Prepare or run the frozen binary and four-choice out-of-domain evaluation."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from customer_policy_verifier.training import ood_choice as ood
from customer_policy_verifier.training.backbone_download import sha256
from customer_policy_verifier.training.train_jev import (
    DEFAULT_BACKBONE, amp_context, build_model, collate, reload_checkpoint, save_json, tokenizer_for,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def predict(model, raw, encoded, tokenizer, cfg, progress):
    import torch
    model.eval()
    results = {}
    # Reduce padding while retaining exactly the frozen order in the report.
    indices = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["input_ids"]))
    for start in range(0, len(indices), cfg["eval_batch_size"]):
        chunk = indices[start:start + cfg["eval_batch_size"]]
        batch = collate([encoded[i] for i in chunk], tokenizer.pad_token_id, cfg["device"])
        with torch.inference_mode(), amp_context(cfg):
            probabilities = model(batch)["logits"].float().softmax(-1).cpu().tolist()
        for i, probabilities_row in zip(chunk, probabilities):
            row = raw[i]
            results[i] = {"id": row["id"], "label_id": row["label_id"],
                "probabilities": probabilities_row, "prompt_sha256": encoded[i]["prompt_sha256"],
                **{k: row[k] for k in ("base_id", "order") if k in row}}
        if start % (cfg["eval_batch_size"] * 20) == 0:
            progress(min(start + len(chunk), len(indices)), len(indices))
    progress(len(indices), len(indices))
    return [results[i] for i in range(len(raw))]


def run(args, cfg, tokenizer):
    import torch
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh output directory")
    manifest, files = ood.load_suite(args.suite, tokenizer)
    plan = json.loads(args.model_plan.read_text())
    if set(plan) != {"base", "original", "mixed"}:
        raise ValueError("Compare the base, original and mixed models")
    # Verify all three weight identities before the first prediction.
    for name, spec in plan.items():
        if name == "base":
            weights = args.backbone / "model.safetensors"
        else:
            weights = Path(spec["checkpoint"])
            selection = json.loads((weights.parent / "selection.json").read_text())
            if selection["checkpoint_sha256"] != spec["sha256"]:
                raise ValueError("Frozen selected checkpoint differs from model plan")
        if sha256(weights) != spec["sha256"]:
            raise ValueError(f"Weight hash differs from model plan: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    execution = {"started_at": now(), "suite_manifest_sha256": sha256(args.suite / "manifest.json"),
        "model_plan": plan, "config": cfg, "suite": manifest,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
            [Path(__file__), ROOT / "src/customer_policy_verifier/training/ood_choice.py",
             ROOT / "src/customer_policy_verifier/training/train_jev.py",
             ROOT / "src/customer_policy_verifier/training/jev_model.py"]},
        "gpu": torch.cuda.get_device_name() if cfg["device"] == "cuda" else None,
        "torch": torch.__version__}
    save_json(args.output / "execution.json", execution)
    status = {"state": "running", "started_at": now()}
    summary = {}
    try:
        for name, spec in plan.items():
            status.update(model=name, stage="loading", completed_questions=0)
            save_json(args.output / "status.json", status)
            started = time.monotonic()
            if name == "base":
                model = build_model(args, cfg)
            else:
                model, saved_cfg, checkpoint = reload_checkpoint(Path(spec["checkpoint"]), args)
                if (saved_cfg.get("input_normalization") != "lowercase"
                        or saved_cfg["num_train_epochs"] != 1 or checkpoint["tiny_model"]):
                    raise ValueError("Only the frozen one-epoch lowercase checkpoints are eligible")
                model.causal_lm.gradient_checkpointing_disable()
            summary[name] = {}
            for task, n in (("ruletaker", 2), ("logiqa2", 4)):
                predictions = {}
                for kind in ("primary", "permuted"):
                    raw, encoded = files[f"{task}-{kind}.jsonl"]
                    status.update(stage=f"{task}_{kind}")

                    def progress(done, total):
                        status.update(completed_questions=done, total_questions=total, updated_at=now())
                        save_json(args.output / "status.json", status)

                    predictions[kind] = predict(model, raw, encoded, tokenizer, cfg, progress)
                report = {"model": name, "task": task, "weight_sha256": spec["sha256"],
                    "suite_manifest_sha256": execution["suite_manifest_sha256"],
                    "metrics": ood.metrics(predictions["primary"], n),
                    "permutation_diagnostic": ood.permutation_summary(predictions["primary"], predictions["permuted"], n),
                    "predictions": predictions}
                save_json(args.output / f"{name}-{task}.json", report)
                summary[name][task] = {k: report[k] for k in ("metrics", "permutation_diagnostic")}
                save_json(args.output / "comparison.json", summary)
                print(json.dumps({"model": name, "task": task, "metrics": report["metrics"]}), flush=True)
            summary[name]["seconds"] = time.monotonic() - started
            del model
            gc.collect()
            if cfg["device"] == "cuda":
                torch.cuda.empty_cache()
        save_json(args.output / "comparison.json", summary)
        status.update(state="complete", completed_at=now())
        save_json(args.output / "status.json", status)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}", updated_at=now())
        save_json(args.output / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--suite", type=Path, default=ROOT / "data/evaluation/ood-choice-v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ood-choice-v1.json")
    parser.add_argument("--model-plan", type=Path, default=ROOT / "configs/ood-models-autodl-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/evaluation/ood-choice-v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    cfg.update(device=args.device, precision=args.precision, eval_batch_size=args.eval_batch_size,
               gradient_checkpointing=False)
    if args.eval_batch_size < 1:
        raise ValueError("Positive batch size required")
    tokenizer = tokenizer_for(args, cfg)
    if args.command == "prepare":
        print(json.dumps(ood.prepare(ROOT, args.suite, tokenizer, cfg["max_tokens"], cfg["seed"]), indent=2))
    else:
        run(args, cfg, tokenizer)


if __name__ == "__main__":
    main()
