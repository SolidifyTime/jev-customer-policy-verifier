"""Archive the complete public Task 2 release and prepare leakage-free original splits.

No model API or credential is used. Pass --download to obtain missing public files;
without it all checks and exports are offline. The release is pinned by SHA-256.
"""

import argparse
import collections
import hashlib
import json
import re
import subprocess
import urllib.request
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT, resolve_path
from customer_policy_verifier.data.prepare_data import RAW, SEED, SPLIT, build_split, load_rows


DATA = ROOT / "data/source/task2"
PUBLICATION = "https://www.amazon.science/publications/tod-procbench-benchmarking-complex-instruction-following-in-task-oriented-dialogues"
ROOT_FOLDER = "https://drive.google.com/drive/folders/12n3z9N1LX3hTNruvKtVaK7XEDEg2G-u5"
TASK_FOLDER = "https://drive.google.com/drive/folders/1aNzCjbYNKUvesqaygrUODV_RlZh0vpTs"
SNAPSHOT = "official-drive-snapshot-2026-10-02"
# Names, IDs and hashes of every JSON in the official Task 2 folder on 2026-10-02.
FILES = (
    ("test_task2_ar_f1.json", "1WyRc954qey8ziPIMLPGLCo-62X6TyaoF", "f3bf939e8d443433d0fbf978e94f28b0ac5938b0c9941966bbf4f3de0f3b3c2f"),
    ("test_task2_de_f1.json", "1CMtRH0givgOVEif45yGaI1-Xp_mm836w", "9c599bf2d06eb933c2337036ab3302a882e1bf16553b849e12f583913bf57f00"),
    ("test_task2_en_f1.json", "1itHuTgGiY5J_WOoANAxJIoj_3rqa0ulY", "5e5402bafdae2ed2c63d6688e161c9e4af8028f596da30dd26b33a3544846b17"),
    ("test_task2_en_f2.json", "13fIMDROue1YBGcwjPzdcJXMn1QH_KOi1", "e415f92cc3aafee76ba22cda10a0046fff2b38573e8f02b9d17127e0e178155e"),
    ("test_task2_en_f3.json", "15Mw0_x_PEYWPyAgEzBpkHnLukGyX5mES", "f63c15a3386d378bb467ae357b3225f98a8216304770ce328f310abb7c8b538e"),
    ("test_task2_es_f1.json", "15OwcfqP_NW8c7hJfyiicBTAFt4d0URWf", "e62be14d2cce19a8fbd9079e428d660d6aa7e6cb673db4dc80c9f4f453814632"),
    ("test_task2_fr_f1.json", "1OGOKGpfWuuFpiPIoswD-3iQjJ1YnwUIa", "b0183bbaefdc71c3add09bbb365b7b54ce88ce5caf366aa6f0d766b6acf7d80d"),
    ("test_task2_hi_f1.json", "1NNfceliUA-8Z28xciMaDy66wVMp-ksp5", "47d35b3559f67c5d02e6c85f4c43d4da1fcd78fbc45c74019f2bc33d21bded6d"),
    ("test_task2_zh_f1.json", "1b8j7ccoh_9esJZLlNUOoQHYA6_sOnTmo", "510087f41601a8a2950bb4341bf9e2415de2494efead9d42de39a1a4eac64617"),
)
METADATA_FILES = (
    ("README.md", "11MsZxNvmJ5BnXsBMIWNnGRYlJ8mHvO2Y", "98e0e8135b1eba99c00527c580222b120a6bd0c846be6fc9a98a754c7ce0526d"),
    ("License", "1bybbv9uA4h-UhYAYI8irNglmqDKfuHrZ", "038ea79c737415a933a62ace42cae213828e7e211935c374310d9a7fcc2ef18a"),
)
EXCLUSIONS = ROOT / "data/archive/reviews/manual-review-20261002/baseline_exclusions.jsonl"
TURN_PREFIX = re.compile(r"^turn \d+: agent: ")


def sha256(path: Path) -> str:
    return hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()


def serialized(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def freeze(path: Path, data: bytes) -> None:
    """Idempotent output that refuses silent modification of frozen artifacts."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen file differs; refusing to overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def check_remote_inventory() -> None:
    with urllib.request.urlopen(TASK_FOLDER, timeout=30) as response:
        page = response.read().decode("utf-8")
    match = re.search(r"window\['_DRIVE_ivd'\] = '(.*?)';", page)
    if not match:
        raise ValueError("Cannot parse the official folder inventory; use offline mode for the pinned release")
    encoded = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m[1], 16)), match[1])
    inventory = json.loads(encoded.replace(r"\/", "/"))[0]
    observed = {(row[2], row[0]) for row in inventory}
    expected = {(name, file_id) for name, file_id, _ in FILES}
    if observed != expected:
        raise ValueError("Official Task 2 inventory changed; review the new release before changing the pinned catalog")


def obtain(path: Path, file_id: str, expected_hash: str, download: bool) -> dict:
    path = resolve_path(path)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}; run with --download")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            subprocess.run([
                "curl", "--fail", "--silent", "--show-error", "--location",
                "--connect-timeout", "15", "--max-time", "120", "--retry", "2",
                "--output", str(temporary), url,
            ], check=True)
            if sha256(temporary) != expected_hash:
                raise ValueError(f"Official file changed or download is incomplete: {path.name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    digest = sha256(path)
    if digest != expected_hash:
        raise ValueError(f"Local archive differs from pinned official file: {path}")
    return {"name": path.name, "path": str(path.resolve()), "drive_id": file_id,
            "download_url": url, "sha256": digest, "bytes": path.stat().st_size,
            "version": SNAPSHOT}


def audit_views(views: dict[str, list[dict]]) -> dict:
    base = views["test_task2_en_f1.json"]
    def fingerprint(row):
        return json.dumps({key: value for key, value in row.items() if key != "instruction"},
                          sort_keys=True, ensure_ascii=False)
    content = collections.Counter(fingerprint(row) for row in base)
    result = {}
    for name, rows in views.items():
        if len(rows) != len(base):
            raise ValueError(f"View has a different number of pairs: {name}")
        if "_en_" in name:
            same = collections.Counter(fingerprint(row) for row in rows) == content
            if not same:
                raise ValueError(f"English instruction format introduces different source samples: {name}")
            result[name] = {"alignment": "multiset of all fields except instruction",
                            "same_base_pairs": True, "new_base_pairs": 0}
        else:
            # The release preserves ordering, English rules and source IDs across translations.
            same = all(a["orig_conv_id"] == b["orig_conv_id"]
                       and a["intent"] == b["intent"]
                       and a["instruction"] == b["instruction"]
                       and len(a["conversation"]) == len(b["conversation"])
                       for a, b in zip(base, rows))
            if not same:
                raise ValueError(f"Translation does not align with the canonical source: {name}")
            result[name] = {"alignment": "row order, orig_conv_id, intent, instruction, history length",
                            "same_base_pairs": True, "new_base_pairs": 0,
                            "translation_semantics_audited": False}
    return result


def split_data(rows: list[dict], raw_hash: str) -> dict:
    expected = build_split(rows, raw_hash)
    split_path = resolve_path(SPLIT)
    if split_path.exists():
        frozen = json.loads(split_path.read_bytes())
        if frozen != expected:
            raise ValueError("Full release differs from the historical frozen split; review before extending")
        return frozen
    return expected


def read_exclusions(rows: list[dict]) -> tuple[set[tuple[int, str]], list[dict]]:
    exclusions_path = resolve_path(EXCLUSIONS)
    if not exclusions_path.exists():
        # These two decisions were completed before the full release archive was prepared.
        records = [
            {"source_row_index": 1028, "expected_label": "violating", "label_supported": "否"},
            {"source_row_index": 1177, "expected_label": "compliant", "label_supported": "无法确定"},
        ]
    else:
        records = [json.loads(line) for line in exclusions_path.read_text().splitlines() if line.strip()]
    keys = {(item["source_row_index"], item["expected_label"]) for item in records}
    if keys != {(1028, "violating"), (1177, "compliant")}:
        raise ValueError("Historical source exclusions changed; review the generation selection")
    if any(not 0 <= index < len(rows) for index, _ in keys):
        raise ValueError("Source exclusion index is outside the canonical release")
    return keys, records


def verify_existing(data: Path = DATA, *, download: bool = False) -> dict:
    """Validate a frozen archive in place without rewriting historical paths.

    The manifest and original JSONL records retain their creation-time bytes.
    Relocated references are resolved only when opening a file, so a package
    move never silently replaces the provenance or its recorded checksums.
    """
    data = resolve_path(data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["manifest_version"] != "task2-full-v1" or manifest["version"] != SNAPSHOT:
        raise ValueError("Unexpected frozen Task 2 archive version")
    views = {}
    for key, catalog in (("files", FILES), ("metadata_files", METADATA_FILES)):
        entries = manifest[key]
        by_name = {entry["name"]: entry for entry in entries}
        if len(by_name) != len(entries) or set(by_name) != {name for name, _, _ in catalog}:
            raise ValueError(f"Frozen archive has an unexpected {key} inventory")
        for name, file_id, expected_hash in catalog:
            entry = by_name[name]
            if entry["sha256"] != expected_hash or entry["drive_id"] != file_id:
                raise ValueError(f"Frozen archive catalog differs from the pinned release: {name}")
            path = resolve_path(entry["path"])
            info = obtain(path, file_id, expected_hash, download)
            if info["bytes"] != entry["bytes"]:
                raise ValueError(f"Frozen archive size changed: {name}")
            if key == "files":
                rows = load_rows(path)
                if (entry["pairs"] != len(rows) or entry["candidates"] != 2 * len(rows)
                        or entry["groups"] != len({row["orig_conv_id"] for row in rows})
                        or entry["intents"] != len({row["intent"] for row in rows})):
                    raise ValueError(f"Frozen archive statistics changed: {name}")
                views[name] = rows
    if audit_views(views) != manifest["view_alignment"]:
        raise ValueError("Frozen archive view alignment changed")
    source_path, split_path = resolve_path(manifest["source_file"]), resolve_path(manifest["split_file"])
    rows = load_rows(source_path)
    source_hash = sha256(source_path)
    if source_hash != manifest["source_sha256"] or source_hash != dict((n, h) for n, _, h in FILES)[source_path.name]:
        raise ValueError("Frozen canonical source changed")
    if sha256(split_path) != manifest["split_sha256"]:
        raise ValueError("Frozen split changed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split != split_data(rows, source_hash):
        raise ValueError("Frozen split differs from the historical group assignment")
    excluded, exclusion_records = read_exclusions(rows)
    originals = {}
    for partition, stored_path in manifest["originals"].items():
        path = resolve_path(stored_path)
        info = manifest["originals_info"][partition]
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Frozen original {partition} data changed")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = {(index, label) for index, row in enumerate(rows)
                    if split["group_assignment"][str(row["orig_conv_id"])] == partition
                    for label in ("compliant", "violating")}
        observed = []
        for record in records:
            meta = record["metadata"]
            index, label = meta["source_row_index"], record["label"]
            if type(index) is not int or not 0 <= index < len(rows) or label not in {"compliant", "violating"}:
                raise ValueError("Invalid frozen original source index or label")
            source = rows[index]
            field = "orig_gt_response" if label == "compliant" else "manipulated_gt_response"
            if (record["id"] != f"original-en_f1-{index}-{label}"
                    or record["policy"] != source["instruction"]
                    or record["conversation"] != source["conversation"]
                    or record["candidate_response"] != source[field]
                    or meta["split"] != partition or meta["source_type"] != "original"
                    or meta["orig_conv_id"] != source["orig_conv_id"] or meta["intent"] != source["intent"]
                    or meta["source_sha256"] != source_hash
                    or resolve_path(meta["source_file"]) != source_path):
                raise ValueError(f"Frozen original source or partition mismatch: {record['id']}")
            observed.append((index, label))
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError(f"Frozen original {partition} has missing or duplicate source labels")
        if (info["candidates"] != len(records)
                or info["groups"] != len({r["metadata"]["orig_conv_id"] for r in records})
                or info["label_counts"] != dict(collections.Counter(r["label"] for r in records))):
            raise ValueError(f"Frozen original {partition} statistics changed")
        originals[partition] = records
    if set(originals) != {"train", "validation", "test"}:
        raise ValueError("Frozen archive must contain train, validation and test originals")
    candidates, reasons, omissions = collections.Counter(), collections.Counter(), []
    for record in originals["train"]:
        index, label = record["metadata"]["source_row_index"], record["label"]
        reason = ("missing_agent_turn_prefix" if not TURN_PREFIX.match(record["candidate_response"]) else
                  "historical_source_label_exclusion" if (index, label) in excluded else None)
        if reason:
            reasons[reason] += 1
            omissions.append({"source_row_index": index, "expected_label": label, "reason": reason})
        else:
            candidates[label] += 1
    eligibility = manifest["generation_eligibility"]
    omission_path = resolve_path(eligibility["exclusions_file"])
    if sha256(omission_path) != eligibility["exclusions_sha256"]:
        raise ValueError("Frozen generation exclusions changed")
    omission = json.loads(omission_path.read_text(encoding="utf-8"))
    if (omission != {"historical_decisions": exclusion_records, "excluded_candidates": omissions}
            or eligibility["original_candidates"] != len(originals["train"])
            or eligibility["eligible_candidates"] != sum(candidates.values())
            or eligibility["label_counts"] != dict(candidates)
            or eligibility["excluded_by_reason"] != dict(reasons)):
        raise ValueError("Frozen generation eligibility audit changed")
    scope = manifest["scope"]
    if (scope["official_task2_json_files"] != len(FILES) or scope["unique_source_pairs"] != len(rows)
            or scope["unique_candidate_samples"] != len(rows) * 2
            or scope["groups"] != len(split["group_assignment"])
            or scope["intents"] != len({row["intent"] for row in rows})):
        raise ValueError("Frozen Task 2 scope statistics changed")
    return manifest


def prepare(data: Path = DATA, download: bool = False) -> dict:
    data = resolve_path(data)
    if download:
        check_remote_inventory()
    if (data / "manifest.json").exists():
        return verify_existing(data, download=download)
    files, views = [], {}
    for name, file_id, digest in FILES:
        path = data / "raw" / name
        info = obtain(path, file_id, digest, download)
        rows = load_rows(path)
        language, instruction_format = name.removesuffix(".json").split("_")[-2:]
        info.update(language=language, instruction_format=instruction_format,
                    pairs=len(rows), candidates=2 * len(rows),
                    groups=len({row["orig_conv_id"] for row in rows}),
                    intents=len({row["intent"] for row in rows}))
        files.append(info)
        views[name] = rows
    metadata = [obtain(data / name, file_id, digest, download)
                for name, file_id, digest in METADATA_FILES]
    alignment = audit_views(views)
    source = data / "raw/test_task2_en_f1.json"
    raw_hash = sha256(source)
    if RAW.exists() and sha256(RAW) != raw_hash:
        raise ValueError("Canonical archive differs from the historical pilot source")
    rows = views[source.name]
    split = split_data(rows, raw_hash)
    split_path = data / "splits" / f"en_f1_seed_{SEED}.json"
    freeze(split_path, serialized(split))
    excluded, exclusion_records = read_exclusions(rows)
    originals = {name: [] for name in ("train", "validation", "test")}
    candidates, reasons = collections.Counter(), collections.Counter()
    omissions = []
    for index, row in enumerate(rows):
        partition = split["group_assignment"][str(row["orig_conv_id"])]
        for label, field in (("compliant", "orig_gt_response"), ("violating", "manipulated_gt_response")):
            record = {
                "id": f"original-en_f1-{index}-{label}", "policy": row["instruction"],
                "conversation": row["conversation"], "candidate_response": row[field], "label": label,
                "metadata": {"source_type": "original", "split": partition,
                             "source_row_index": index, "orig_conv_id": row["orig_conv_id"],
                             "intent": row["intent"], "source_file": str(source),
                             "source_sha256": raw_hash, "language": "en", "instruction_format": "f1",
                             "label_origin": "original_benchmark", "review_status": "original_source_label"},
            }
            originals[partition].append(record)
            if partition == "train":
                reason = ("missing_agent_turn_prefix" if not TURN_PREFIX.match(row[field]) else
                          "historical_source_label_exclusion" if (index, label) in excluded else None)
                if reason:
                    reasons[reason] += 1
                    omissions.append({"source_row_index": index, "expected_label": label, "reason": reason})
                else:
                    candidates[label] += 1
    original_paths, original_info = {}, {}
    for partition, records in originals.items():
        path = data / "originals" / f"{partition}.jsonl"
        freeze(path, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records).encode())
        original_paths[partition] = str(path)
        original_info[partition] = {"sha256": sha256(path), "candidates": len(records),
                                   "groups": len({item["metadata"]["orig_conv_id"] for item in records}),
                                   "label_counts": dict(collections.Counter(item["label"] for item in records))}
    omission_path = data / "generation_exclusions.json"
    freeze(omission_path, serialized({"historical_decisions": exclusion_records, "excluded_candidates": omissions}))
    manifest = {
        "manifest_version": "task2-full-v1", "dataset": "TOD-ProcBench Task 2", "version": SNAPSHOT,
        "publication_url": PUBLICATION, "root_folder_url": ROOT_FOLDER, "task_folder_url": TASK_FOLDER,
        "license": "Llama 3.3 Community License Agreement (upstream release)",
        "source_file": str(source), "source_sha256": raw_hash,
        "split_file": str(split_path), "split_sha256": sha256(split_path),
        "originals": original_paths, "originals_info": original_info,
        "files": files, "metadata_files": metadata, "view_alignment": alignment,
        "scope": {"official_task2_json_files": len(files), "canonical_language": "en",
                  "canonical_instruction_format": "f1", "unique_source_pairs": len(rows),
                  "unique_candidate_samples": len(rows) * 2, "groups": len(split["group_assignment"]),
                  "intents": len({row["intent"] for row in rows}),
                  "noncanonical_views_used_as_extra_training_sources": False},
        "split_policy": {"kind": "custom_group_split", "group_key": "orig_conv_id", "seed": SEED,
                         "target_ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
                         "historical_assignment_preserved": True,
                         "note": "Public files are named test; these are project-specific train/validation/test partitions."},
        "generation_eligibility": {"partition": "train", "original_candidates": len(originals["train"]),
                                   "eligible_candidates": sum(candidates.values()),
                                   "label_counts": dict(candidates), "excluded_by_reason": dict(reasons),
                                   "exclusions_file": str(omission_path), "exclusions_sha256": sha256(omission_path)},
    }
    freeze(data / "manifest.json", serialized(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Verify official inventory and download missing pinned public files")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--verify-existing", action="store_true", help="Read and verify the frozen archive without rewriting provenance")
    args = parser.parse_args()
    if args.verify_existing and args.download:
        parser.error("--verify-existing is offline; omit --download")
    manifest = verify_existing(args.data_dir) if args.verify_existing else prepare(args.data_dir, args.download)
    print(json.dumps({"manifest": str(resolve_path(args.data_dir) / "manifest.json"),
                      "scope": manifest["scope"], "originals_info": manifest["originals_info"],
                      "generation_eligibility": manifest["generation_eligibility"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
