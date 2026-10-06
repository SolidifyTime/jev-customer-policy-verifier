"""Merge original and reviewed-export synthetic records into reproducible training JSONL."""

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT, resolve_path
from customer_policy_verifier.data.prepare_data import SEED, load_rows
from customer_policy_verifier.data.prepare_full_data import freeze


VERSION = "training-data-v1"
MODEL_FIELDS = ("policy", "conversation", "candidate_response")
FIELDS = {"id", *MODEL_FIELDS, "label", "metadata"}


def read_json(path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in resolve_path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()


def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def fingerprint(row):
    values = {key: ([" ".join(turn.split()) for turn in row[key]] if isinstance(row[key], list)
                    else " ".join(row[key].split())) for key in MODEL_FIELDS}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def validate_record(record, partition, source_type, raw, split, source_hash):
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError("Unexpected training record fields")
    if record["label"] not in {"compliant", "violating"}:
        raise ValueError("Unknown label")
    for key in ("id", "policy", "candidate_response"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"Empty or invalid {key}")
    turns = record["conversation"]
    if not isinstance(turns, list) or not turns or any(not isinstance(t, str) or not t.strip() for t in turns):
        raise ValueError("Missing conversation turns")
    meta = record["metadata"]
    index = meta["source_row_index"]
    if type(index) is not int or not 0 <= index < len(raw):
        raise ValueError("Invalid source row index")
    original = raw[index]
    if (meta["split"] != partition or meta["source_type"] != source_type
            or meta["source_sha256"] != source_hash or meta["orig_conv_id"] != original["orig_conv_id"]
            or split["group_assignment"][str(meta["orig_conv_id"])] != partition
            or meta["intent"] != original["intent"] or record["policy"] != original["instruction"]):
        raise ValueError(f"Source or partition mismatch: {record['id']}")
    if source_type == "original":
        field = "orig_gt_response" if record["label"] == "compliant" else "manipulated_gt_response"
        if record["conversation"] != original["conversation"] or record["candidate_response"] != original[field]:
            raise ValueError(f"Original source content changed: {record['id']}")
    elif (partition != "train" or meta.get("batch_spot_check") not in {"completed", "completed_with_issues"}
          or meta.get("review_status") not in {"spot_checked", "not_individually_reviewed"}):
        raise ValueError(f"Synthetic record has no completed batch spot check: {record['id']}")


def unique_records(rows):
    kept, duplicates, seen, ids = [], [], {}, set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"Duplicate record ID: {row['id']}")
        ids.add(row["id"])
        key = fingerprint(row)
        if key in seen:
            original = seen[key]
            if original["label"] != row["label"]:
                raise ValueError(f"Same input has conflicting labels: {original['id']} / {row['id']}")
            duplicates.append({"removed_id": row["id"], "kept_id": original["id"]})
        else:
            seen[key] = row
            kept.append(row)
    return kept, duplicates


def stats(rows):
    return {"count": len(rows), "label_counts": dict(sorted(collections.Counter(r["label"] for r in rows).items())),
            "source_type_counts": dict(sorted(collections.Counter(r["metadata"]["source_type"] for r in rows).items())),
            "review_status_counts": dict(sorted(collections.Counter(r["metadata"]["review_status"] for r in rows).items())),
            "conversation_groups": len({r["metadata"]["orig_conv_id"] for r in rows}),
            "intents": len({r["metadata"]["intent"] for r in rows})}


def verify_existing(output):
    """Check recorded inputs and outputs while retaining the original manifest."""
    output = resolve_path(output)
    manifest = read_json(output / "manifest.json")
    if (manifest["version"] != VERSION or manifest["model_input_fields"] != list(MODEL_FIELDS)
            or manifest["target_field"] != "label"
            or manifest["label_to_id"] != {"compliant": 0, "violating": 1}):
        raise ValueError("Unexpected frozen training bundle version or model schema")
    for item in manifest["inputs"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen training input changed: {resolve_path(item['path'])}")
    if set(manifest["files"]) != {"train.jsonl", "train_original.jsonl", "validation.jsonl", "test.jsonl"}:
        raise ValueError("Unexpected frozen training file inventory")
    for name, info in manifest["files"].items():
        path = output / name
        if sha256(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Frozen training file changed: {path}")
        if any(info[key] != value for key, value in stats(read_jsonl(path)).items()):
            raise ValueError(f"Frozen training file statistics changed: {path}")
    return manifest


def build(dataset_manifest, synthetic_path, exclusions_path, output, seed):
    dataset_manifest, synthetic_path, exclusions_path, output = map(
        resolve_path, (dataset_manifest, synthetic_path, exclusions_path, output))
    dataset = read_json(dataset_manifest)
    raw_path, split_path = resolve_path(dataset["source_file"]), resolve_path(dataset["split_file"])
    source_hash = sha256(raw_path)
    raw, split = load_rows(raw_path), read_json(split_path)
    if (source_hash != dataset["source_sha256"] or source_hash != split["source_sha256"]
            or sha256(split_path) != dataset["split_sha256"]):
        raise ValueError("Frozen raw source or split changed")

    original, input_paths = {}, [dataset_manifest, raw_path, split_path, synthetic_path, exclusions_path]
    for partition, name in dataset["originals"].items():
        path = resolve_path(name)
        if sha256(path) != dataset["originals_info"][partition]["sha256"]:
            raise ValueError(f"Original {partition} file changed")
        original[partition] = read_jsonl(path)
        input_paths.append(path)
        for row in original[partition]:
            validate_record(row, partition, "original", raw, split, source_hash)

    synthetic = read_jsonl(synthetic_path)
    for row in synthetic:
        validate_record(row, "train", "synthetic", raw, split, source_hash)
    export_manifest_path = synthetic_path.parent / "manifest.json"
    export_summary_path = synthetic_path.parent / "summary.json"
    exported, export_manifest = read_json(export_summary_path), read_json(export_manifest_path)
    if (export_manifest["source_sha256"] != source_hash
            or export_manifest["split_sha256"] != dataset["split_sha256"]
            or len(synthetic) != exported["exported_count"]
            or dict(collections.Counter(r["label"] for r in synthetic)) != exported["exported_label_counts"]):
        raise ValueError("Synthetic export does not match its completed summary or source")
    input_paths += [export_manifest_path, export_summary_path]

    decisions = read_jsonl(exclusions_path)
    exclusions = {(r["source_row_index"], r["expected_label"]) for r in decisions}
    available = {(r["metadata"]["source_row_index"], r["label"]) for r in original["train"]}
    if not exclusions <= available or len(exclusions) != len(decisions):
        raise ValueError("Exclusions contain duplicate or non-training source labels")
    def excluded(row):
        return (row["metadata"]["source_row_index"], row["label"]) in exclusions
    baseline = [r for r in original["train"] if not excluded(r)]
    removed_synthetic = [r["id"] for r in synthetic if excluded(r)]
    baseline, baseline_duplicates = unique_records(baseline)
    train, merge_duplicates = unique_records([*baseline, *[r for r in synthetic if not excluded(r)]])
    # Mixing is deterministic, and each record retains its existing source and review metadata.
    train.sort(key=lambda r: r["id"])
    random.Random(seed).shuffle(train)

    partitions = {"train": train, "validation": original["validation"], "test": original["test"]}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if ({r["metadata"]["orig_conv_id"] for r in partitions[a]}
                & {r["metadata"]["orig_conv_id"] for r in partitions[b]}):
            raise ValueError(f"Conversation groups cross partitions: {a} / {b}")
        if {fingerprint(r) for r in partitions[a]} & {fingerprint(r) for r in partitions[b]}:
            raise ValueError(f"Identical full inputs cross partitions: {a} / {b}")

    payloads = {"train.jsonl": jsonl_bytes(train), "train_original.jsonl": jsonl_bytes(baseline),
                "validation.jsonl": resolve_path(dataset["originals"]["validation"]).read_bytes(),
                "test.jsonl": resolve_path(dataset["originals"]["test"]).read_bytes()}
    if (output / "manifest.json").exists():
        manifest = verify_existing(output)
        expected_inputs = [(resolve_path(path), sha256(path)) for path in input_paths]
        recorded_inputs = [(resolve_path(item["path"]), item["sha256"]) for item in manifest["inputs"]]
        if (recorded_inputs != expected_inputs or manifest["seed"] != seed
                or manifest["source_sha256"] != source_hash
                or manifest["split_sha256"] != dataset["split_sha256"]
                or manifest["excluded_source_labels"] != decisions
                or manifest["excluded_synthetic_ids"] != removed_synthetic
                or manifest["duplicates_removed"] != baseline_duplicates + merge_duplicates):
            raise ValueError("Existing training bundle uses different inputs or settings; choose a new output directory")
        for name, data in payloads.items():
            if (output / name).read_bytes() != data:
                raise ValueError(f"Existing training bundle differs from its frozen inputs: {name}")
        # The historical builder hash remains provenance. A package relocation
        # does not authorize replacing it with today's implementation hash.
        print(json.dumps({"output_directory": str(output), "status": "verified_existing",
                          "files": manifest["files"]}, ensure_ascii=False, indent=2))
        return manifest
    manifest = {"version": VERSION, "seed": seed, "model_input_fields": list(MODEL_FIELDS),
                "target_field": "label", "label_to_id": {"compliant": 0, "violating": 1},
                "metadata_usage": "Provenance only; never include id, label or metadata in model input.",
                "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in input_paths],
                "source_sha256": source_hash, "split_sha256": dataset["split_sha256"],
                "builder_sha256": sha256(Path(__file__)),
                "excluded_source_labels": decisions, "excluded_synthetic_ids": removed_synthetic,
                "duplicates_removed": baseline_duplicates + merge_duplicates,
                "checks": {"source_partitions_validated": True, "no_group_overlap": True,
                           "no_cross_partition_exact_input_overlap": True, "no_conflicting_labels_for_same_input": True},
                "files": {name: {**stats(records), "sha256": hashlib.sha256(payloads[name]).hexdigest(),
                                  "bytes": len(payloads[name])} for name, records in
                          (("train.jsonl", train), ("train_original.jsonl", baseline),
                           ("validation.jsonl", original["validation"]), ("test.jsonl", original["test"]))}}
    payloads["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    # Preflight every destination before writing; existing, different datasets are never overwritten.
    for name, data in payloads.items():
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Output differs; choose a new output directory: {path}")
    for name, data in payloads.items():
        freeze(output / name, data)
    print(json.dumps({"output_directory": str(output), "files": manifest["files"],
                      "excluded_source_labels": len(exclusions),
                      "duplicates_removed": len(manifest["duplicates_removed"])}, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, default=ROOT / "data/source/task2/manifest.json")
    parser.add_argument("--synthetic", type=Path, default=ROOT / "data/derived/bulk-openrouter-20261003-v3/training_variants.jsonl")
    parser.add_argument("--exclusions", type=Path, default=ROOT / "data/archive/reviews/manual-review-20261002/baseline_exclusions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/training/task2-v1")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verify-existing", action="store_true", help="Require and verify the frozen bundle without replacing its provenance")
    args = parser.parse_args()
    if args.verify_existing and not (resolve_path(args.output_dir) / "manifest.json").exists():
        parser.error("--verify-existing requires an existing output manifest")
    build(args.dataset_manifest, args.synthetic, args.exclusions, args.output_dir, args.seed)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
