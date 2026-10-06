"""Validate TOD-ProcBench Task 2 and freeze a conversation-level split."""

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT, resolve_path

RAW = ROOT / "data/archive/pilot-source/raw/test_task2_en_f1.json"
SPLIT = ROOT / "data/archive/pilot-source/splits/en_f1_seed_20261002.json"
SEED = 20261002
FIELDS = {
    "conversation", "intent", "instruction", "orig_gt_response",
    "orig_conv_id", "manipulated_gt_response",
}


def load_rows(path: Path) -> list[dict]:
    path = resolve_path(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError(f"Unexpected fields in source row {index}")
        if not isinstance(row["orig_conv_id"], int):
            raise ValueError(f"Missing integer conversation ID in row {index}")
        if not isinstance(row["conversation"], list) or not row["conversation"]:
            raise ValueError(f"Missing conversation in row {index}")
        if any(not isinstance(turn, str) or not turn.strip() for turn in row["conversation"]):
            raise ValueError(f"Invalid conversation turn in row {index}")
        for field in FIELDS - {"conversation", "orig_conv_id"}:
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Empty {field} in row {index}")
        if row["orig_gt_response"] == row["manipulated_gt_response"]:
            raise ValueError(f"Identical positive and negative response in row {index}")
    return rows


def build_split(rows: list[dict], raw_hash: str) -> dict:
    groups = sorted({row["orig_conv_id"] for row in rows})
    random.Random(SEED).shuffle(groups)
    train_end = round(len(groups) * 0.6)
    val_end = train_end + round(len(groups) * 0.2)
    assignment = {
        str(group): ("train" if i < train_end else "validation" if i < val_end else "test")
        for i, group in enumerate(groups)
    }
    group_counts = collections.Counter(assignment.values())
    pair_counts = collections.Counter(assignment[str(row["orig_conv_id"])] for row in rows)
    return {
        "source_file": RAW.name,
        "source_sha256": raw_hash,
        "seed": SEED,
        "group_key": "orig_conv_id",
        "source_pairs": len(rows),
        "candidate_samples": len(rows) * 2,
        "group_counts": dict(sorted(group_counts.items())),
        "candidate_label_counts": {"compliant": len(rows), "violating": len(rows)},
        "split_pair_counts": dict(sorted(pair_counts.items())),
        "split_candidate_counts": {key: count * 2 for key, count in sorted(pair_counts.items())},
        "group_assignment": assignment,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--split", type=Path, default=SPLIT)
    args = parser.parse_args()
    args.raw, args.split = resolve_path(args.raw), resolve_path(args.split)
    rows = load_rows(args.raw)
    raw_hash = hashlib.sha256(args.raw.read_bytes()).hexdigest()
    result = build_split(rows, raw_hash)
    if args.split.exists():
        old = json.loads(args.split.read_text(encoding="utf-8"))
        if old != result:
            raise SystemExit("Existing split differs; refusing to overwrite frozen assignment")
        print("Verified existing frozen split:")
    else:
        args.split.parent.mkdir(parents=True, exist_ok=True)
        args.split.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Created frozen split:")
    print(json.dumps({key: value for key, value in result.items() if key != "group_assignment"}, indent=2))


if __name__ == "__main__":
    main()
