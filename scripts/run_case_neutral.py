"""Train both one-epoch case-neutral models, then evaluate frozen selections."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from customer_policy_verifier.training.train_jev import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/jev-training-case-neutral-1epoch.json")
    parser.add_argument("--status-dir", type=Path, default=ROOT / "runs/training/case-neutral-1epoch-v1")
    parser.add_argument("--original-run-dir", type=Path,
                        default=ROOT / "runs/training/llamafactory-direct-original-case-neutral-1epoch-v1")
    parser.add_argument("--mixed-run-dir", type=Path,
                        default=ROOT / "runs/training/llamafactory-direct-mixed-case-neutral-1epoch-v1")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    expected = {"input_normalization": "lowercase", "num_train_epochs": 1, "max_steps": -1,
                "warmup_steps": 0, "warmup_ratio": 0.1, "eval_strategy": "epoch"}
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError("Comparison requires lowercase evidence, one epoch, ratio warmup and epoch evaluation")
    runs = {"original": args.original_run_dir.resolve(), "mixed": args.mixed_run_dir.resolve()}
    if runs["original"] == runs["mixed"]:
        raise ValueError("The two experiments require separate run directories")
    for directory in (*runs.values(), args.status_dir):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Use a fresh experiment directory: {directory}")
    args.status_dir.mkdir(parents=True, exist_ok=True)
    status = {"state": "starting", "started_at": datetime.now(timezone.utc).isoformat(),
              "config": str(args.config.resolve()),
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "runs": {key: str(value) for key, value in runs.items()}}

    def stage(name, command):
        status.update(state="running", stage=name, command=command)
        save_json(args.status_dir / "status.json", status)
        print(f"\nStarting {name}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    common = [sys.executable, "-u", str(ROOT / "scripts/cpv.py"), "train"]
    try:
        # Freeze both selections before reading either test result.
        for name, directory in runs.items():
            stage("training_" + name, [*common, "train", "--config", str(args.config),
                                      "--train-set", name, "--run-dir", str(directory)])
        summary = {}
        for name, directory in runs.items():
            summary[name] = {"selection": json.loads((directory / "selection.json").read_text())}
            for split in ("validation", "test"):
                output = directory / f"{split}-report.json"
                stage(f"evaluating_{name}_{split}", [*common, "evaluate", "--config", str(args.config),
                      "--checkpoint", str(directory / "best.safetensors"), "--split", split, "--output", str(output)])
                summary[name][split] = json.loads(output.read_text())["metrics"]
        save_json(args.status_dir / "comparison.json", summary)
        status.update(state="complete", stage="complete", completed_at=datetime.now(timezone.utc).isoformat())
        save_json(args.status_dir / "status.json", status)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        save_json(args.status_dir / "status.json", status)
        raise


if __name__ == "__main__":
    main()
