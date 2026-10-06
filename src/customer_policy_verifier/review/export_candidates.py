"""Export generated candidates and a tiny spot check; no model API or exhaustive review."""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT, resolve_path
from customer_policy_verifier.generation.pilot import EXTRACT_SCHEMA, GENERATE_SCHEMA, check_extraction, check_variant, schema_issues
from customer_policy_verifier.data.prepare_data import RAW, SEED, SPLIT, load_rows


DEFAULT_EXCLUSIONS = ROOT / "data/archive/reviews/manual-review-20261002/baseline_exclusions.jsonl"


def read_json(path: Path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                    encoding="utf-8")


def known_rejected_generations() -> set[str]:
    """Keep previously rejected/deferred exact artifacts out of a fresh lightweight export."""
    hashes = set()
    review_paths = sorted({path for folder in (ROOT / "runs/generation", ROOT / "runs/archive/bulk-preflight")
                           for path in folder.glob("*/preflight-review.json")})
    for review_path in review_paths:
        for decision in read_json(review_path)["decisions"]:
            if decision["verdict"] == "通过":
                continue
            output = resolve_path(review_path.parent / decision["generation_file"]).resolve()
            if not output.is_relative_to(review_path.parent.resolve() / "sources"):
                raise ValueError("Preflight generation path is outside its run")
            if sha256(output) != decision["generation_sha256"]:
                raise ValueError("Generation changed since the preflight review")
            hashes.add(decision["generation_sha256"])
    manifest = ROOT / "runs/reviews/manual-review-20261002/manifest.json"
    decisions = DEFAULT_EXCLUSIONS.parent / "decisions.json"
    if not manifest.exists() or not decisions.exists():
        return hashes
    rejected = {variant["variant_id"] for source in read_json(decisions)
                for variant in source["variant_reviews"] if variant["verdict"] != "通过"}
    return hashes | {variant["output_sha256"] for scene in read_json(manifest)["scenes"]
                     for variant in scene["variants"] if variant["variant_id"] in rejected}


def block(value: str | list[str]) -> str:
    text = "\n".join(value) if isinstance(value, list) else value
    return f"````text\n{text}\n````"


def validate_sample(sample: dict, rows: list[dict], split: dict) -> None:
    index = sample["source_row_index"]
    if type(index) is not int or not 0 <= index < len(rows):
        raise ValueError("Invalid source row index")
    original = rows[index]
    field = {"compliant": "orig_gt_response", "violating": "manipulated_gt_response"}[
        sample["expected_label"]]
    for key, source_key in (("policy", "instruction"), ("conversation", "conversation"),
                            ("candidate_response", field), ("orig_conv_id", "orig_conv_id"),
                            ("intent", "intent")):
        if sample[key] != original[source_key]:
            raise ValueError(f"Source row {index} differs from frozen raw data: {key}")
    if split["group_assignment"][str(sample["orig_conv_id"])] != "train":
        raise ValueError(f"Source row {index} is outside the training split")


def dataset_paths(selection: dict) -> tuple[Path, Path]:
    if selection.get("kind") != "bulk":
        return RAW, SPLIT
    if not Path(selection["source_file"]).is_absolute() or not Path(selection["split_file"]).is_absolute():
        raise ValueError("Bulk source and split paths must be absolute")
    raw, split = resolve_path(selection["source_file"]), resolve_path(selection["split_file"])
    return raw, split


def bulk_entries(run: Path, selection: dict, summary: dict, split: dict):
    run = resolve_path(run)
    raw_path, split_path = dataset_paths(selection)
    if (sha256(raw_path) != selection["source_sha256"]
            or sha256(split_path) != selection["split_sha256"]
            or selection["source_sha256"] != split["source_sha256"]
            or read_json(split_path) != split):
        raise ValueError("Bulk source or frozen split differs")
    samples, results = selection["samples"], summary["results"]
    if (summary.get("status") != "completed" or summary["completed_count"] != len(samples)
            or len(results) != len(samples)):
        raise ValueError(f"Bulk run is incomplete: {run}")
    source_keys = [(s["source_row_index"], s["expected_label"]) for s in samples]
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("Bulk selection contains duplicate source and label pairs")
    numbers = [r["source_number"] for r in results]
    if (any(type(number) is not int for number in numbers)
            or set(numbers) != set(range(len(samples))) or len(set(numbers)) != len(numbers)):
        raise ValueError("Bulk results contain missing or duplicate source numbers")
    count = selection["variants_per_scene"]
    if type(count) is not int or count < 1:
        raise ValueError("Bulk variants_per_scene must be positive")
    by_number = {result["source_number"]: result for result in results}
    entries = []
    for number, sample in enumerate(samples):
        result = by_number[number]
        if (sample["source_row_index"] != result["source_row_index"]
                or sample["expected_label"] != result["expected_label"]):
            raise ValueError("Selection and result source or label differ")
        folder = run / "sources" / f"{number:05d}"
        extracted = folder / "extract.json"
        if not extracted.exists():
            raise ValueError(f"Missing extraction: {extracted}")
        generations = result["generations"]
        if result["status"] == "rejected_extraction":
            if generations:
                raise ValueError("Rejected extraction has generation results")
            # Keep rejected sources in the common source-validation loop.
            entries.append((sample, result, extracted, None, 0))
            continue
        if result["status"] != "completed":
            raise ValueError(f"Bulk source is incomplete: {number}")
        slots = [g["variant_number"] for g in generations]
        if (len(slots) != count or any(type(slot) is not int for slot in slots)
                or set(slots) != set(range(1, count + 1))):
            raise ValueError(f"Bulk source has missing or duplicate generation slots: {number}")
        for generation in sorted(generations, key=lambda item: item["variant_number"]):
            if generation["status"] not in {
                    "candidate_for_human_review", "rejected_format", "rejected_duplicate"}:
                raise ValueError(f"Bulk generation is incomplete: {number}")
            slot = generation["variant_number"]
            generated = folder / f"generate-{slot:02d}.json"
            if not generated.exists():
                raise ValueError(f"Missing generation: {generated}")
            entries.append((sample, result, extracted, generated, slot))
    return entries


def generation_records(run: Path, rows: list[dict], split: dict):
    run = resolve_path(run)
    selection, summary = read_json(run / "selection.json"), read_json(run / "summary.json")
    if selection["model"] != summary["model"] or selection["prompt_version"] != summary["prompt_version"]:
        raise ValueError(f"Selection and summary configuration differ: {run}")
    results = summary["results"]
    if selection.get("kind") == "bulk":
        entries = bulk_entries(run, selection, summary, split)
    elif "samples" in selection:
        samples = selection["samples"]
        if summary["completed_count"] != len(samples) or len(results) != len(samples):
            raise ValueError(f"Batch is incomplete: {run}")
        if selection["source_sha256"] != split["source_sha256"]:
            raise ValueError("Batch uses another raw source")
        entries = []
        for number, (sample, result) in enumerate(zip(samples, results)):
            generated = run / f"sample-{number}-generate.json"
            if result["status"] == "interrupted":
                raise ValueError(f"Batch contains an interrupted source: {run}")
            if result["status"] != "rejected_extraction" and not generated.exists():
                raise ValueError(f"Missing generation: {generated}")
            entries.append((sample, result, run / f"sample-{number}-extract.json", generated, 1))
    else:
        base = resolve_path(selection["base_run"])
        base_selection, base_summary = read_json(base / "selection.json"), read_json(base / "summary.json")
        if (base_selection["source_sha256"] != split["source_sha256"]
                or base_summary["completed_count"] != len(base_selection["samples"])
                or base_selection["model"] != selection["model"]):
            raise ValueError("Expansion base source, model or completion state differs")
        sources, count = selection["sources"], selection["extra_per_scene"]
        expected = {(number, slot) for number in range(len(sources)) for slot in range(2, count + 2)}
        actual = {(r["source_number"], r["variant_number"]) for r in results}
        if (summary["completed_generations"] != len(expected) or len(results) != len(expected)
                or actual != expected or any(r["status"] == "interrupted" for r in results)):
            raise ValueError(f"Expansion is incomplete or has duplicate results: {run}")
        entries = []
        for result in results:
            number, slot = result["source_number"], result["variant_number"]
            source = sources[number]
            sample = source["sample"]
            extracted = base / f"sample-{source['base_number']}-extract.json"
            if (sample != base_selection["samples"][source["base_number"]]
                    or source["scene"] != read_json(extracted)["output"]):
                raise ValueError("Expansion source differs from base scene")
            entries.append((sample, result, extracted, run / f"sample-{number}-generate-{slot}.json", slot))
    for sample, result, extracted, generated, slot in entries:
        validate_sample(sample, rows, split)
        if (sample["source_row_index"] != result["source_row_index"]
                or sample["expected_label"] != result["expected_label"]):
            raise ValueError("Selection and result source or label differ")
        if generated is not None and generated.exists():
            yield sample, extracted, generated, slot, selection


def select_spot_checks(candidates: list[dict], count: int, seed: int) -> list[str]:
    ranked = sorted(candidates, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
    selected = []
    for label in ("compliant", "violating"):
        choices = [row for row in ranked if row["label"] == label]
        if choices and len(selected) < count:
            selected.append(choices[0])
    while len(selected) < min(count, len(ranked)):
        remaining = [row for row in ranked if row not in selected]
        groups = {row["metadata"]["orig_conv_id"] for row in selected}
        fresh = [row for row in remaining if row["metadata"]["orig_conv_id"] not in groups]
        selected.append((fresh or remaining)[0])
    return [row["id"] for row in selected]


def render_spot_check(candidates: list[dict], selected: list[str], originals: dict) -> str:
    by_id = {row["id"]: row for row in candidates}
    lines = ["# 少量抽查", "", f"本批仅抽查 **{len(selected)} 条**，不需要逐条审核全部候选。", "",
             "只判断新样本本身的预期标签是否成立，以及是否新增会影响判断的事实。",
             "普通措辞、语气或不改变标签的改写差异可以通过；不要求完全等义。",
             "将结论改为“通过／剔除／待定”；有问题时写一句备注，不用润色对话。",
             "未抽中的样本保留未逐条复核标记；少量抽查不代表整批质量合格。", ""]
    for identity in selected:
        row, source = by_id[identity], originals[identity]
        lines += [f"## 候选 {identity}", "",
                  f"源行：{row['metadata']['source_row_index']}；预期标签：**{row['label']}**。", "",
                  "### 规则", "", block(row["policy"]), "",
                  "### 原始历史", "", block(source["conversation"]), "",
                  "### 原始待审核回复", "", block(source["candidate_response"]), "",
                  "### 新历史", "", block(row["conversation"]), "",
                  "### 新待审核回复", "", block(row["candidate_response"]), "",
                  "结论（通过／剔除／待定）：待填写", "", "备注：无", ""]
    return "\n".join(lines)


def prepare(runs: list[Path], output: Path, count: int, seed: int, exclusions_path: Path) -> None:
    runs = [resolve_path(run) for run in runs]
    output, exclusions_path = resolve_path(output), resolve_path(exclusions_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; refusing to overwrite spot-check work")
    raw_path, split_path = dataset_paths(read_json(runs[0] / "selection.json"))
    split, rows = read_json(split_path), load_rows(raw_path)
    source_hash, split_hash = sha256(raw_path), sha256(split_path)
    if source_hash != split["source_sha256"]:
        raise ValueError("Raw source differs from the frozen split")
    for run in runs[1:]:
        next_raw, next_split = dataset_paths(read_json(run / "selection.json"))
        if sha256(next_raw) != source_hash or sha256(next_split) != split_hash:
            raise ValueError("Cannot combine runs from different sources or frozen splits")
    exclusions = set()
    if exclusions_path.exists():
        exclusions = {(row["source_row_index"], row["expected_label"])
                      for row in map(json.loads, exclusions_path.read_text().splitlines())}
    known_rejections = known_rejected_generations()
    candidates, originals, rejected, seen = [], {}, [], {}
    for run in runs:
        for source, extracted_path, generated_path, slot, selection in generation_records(run, rows, split):
            extracted, generated = read_json(extracted_path), read_json(generated_path)
            scene, variant = extracted["output"], generated["output"]
            reasons = extracted["api"]["schema_issues"] + schema_issues(scene, EXTRACT_SCHEMA["json_schema"]["schema"])
            if not reasons:
                reasons = check_extraction(source, scene)
                if not scene["can_expand"]:
                    reasons.append("source cannot expand")
            if not reasons:
                reasons = generated["api"]["schema_issues"] + schema_issues(variant, GENERATE_SCHEMA["json_schema"]["schema"])
            if not reasons:
                reasons = check_variant(source, variant)
            if (source["source_row_index"], source["expected_label"]) in exclusions:
                reasons.append("known unsupported or uncertain source label")
            if sha256(generated_path) in known_rejections:
                reasons.append("this exact generation was previously rejected or deferred")
            if reasons:
                rejected.append({"generation_output": str(generated_path), "reasons": reasons})
                continue
            content = {"policy": source["policy"], "conversation": variant["conversation"],
                       "candidate_response": variant["candidate_response"]}
            normalized = {key: ([" ".join(turn.split()) for turn in value] if isinstance(value, list)
                                else " ".join(value.split())) for key, value in content.items()}
            fingerprint = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
            if fingerprint in seen:
                if seen[fingerprint] != source["expected_label"]:
                    raise ValueError("Identical generated input has conflicting labels")
                rejected.append({"generation_output": str(generated_path), "reasons": ["duplicate candidate"]})
                continue
            seen[fingerprint] = source["expected_label"]
            identity = f"synthetic-{source['source_row_index']}-{fingerprint[:16]}"
            candidates.append({"id": identity, **content, "label": source["expected_label"], "metadata": {
                "source_type": "synthetic", "split": "train", "source_row_index": source["source_row_index"],
                "orig_conv_id": source["orig_conv_id"], "intent": source["intent"], "variant_number": slot,
                "model": selection["model"], "prompt_version": selection["prompt_version"],
                "generation_output": str(generated_path), "generation_sha256": sha256(generated_path),
                "extraction_output": str(extracted_path), "extraction_sha256": sha256(extracted_path),
                "source_sha256": split["source_sha256"], "review_status": "not_individually_reviewed",
                "label_origin": "inherited_from_source", "batch_spot_check": "pending",
            }})
            originals[identity] = source
    if not candidates:
        raise ValueError("No candidates pass program checks and known source-label exclusions")
    selected = select_spot_checks(candidates, count, seed)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "candidates.jsonl", candidates)
    (output / "spot_check.md").write_text(render_spot_check(candidates, selected, originals), encoding="utf-8")
    manifest = {"run_dirs": list(map(str, runs)), "source_sha256": split["source_sha256"],
                "source_file": str(raw_path.resolve()), "split_file": str(split_path.resolve()),
                "split_sha256": split_hash, "candidates_sha256": sha256(output / "candidates.jsonl"),
                "seed": seed, "spot_check_ids": selected, "originals": {key: originals[key] for key in selected}}
    write_json(output / "manifest.json", manifest)
    summary = {"candidate_count": len(candidates), "candidate_label_counts": dict(collections.Counter(r["label"] for r in candidates)),
               "spot_check_count": len(selected), "spot_check_ids": selected, "excluded_count": len(rejected),
               "excluded": rejected, "status": "candidates exported; tiny spot check pending; no training run"}
    write_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("candidate_count", "candidate_label_counts", "spot_check_count", "excluded_count")}, ensure_ascii=False))
    print(f"Spot check: {output / 'spot_check.md'}")


def apply_spot_check(output: Path, reviewer: str) -> None:
    output = resolve_path(output)
    if (output / "training_variants.jsonl").exists() or (output / "spot_check_results.json").exists():
        raise ValueError("Spot check already imported; refusing to overwrite exported results")
    manifest, summary = read_json(output / "manifest.json"), read_json(output / "summary.json")
    raw_path = resolve_path(manifest.get("source_file", RAW))
    split_path = resolve_path(manifest.get("split_file", SPLIT))
    if sha256(raw_path) != manifest["source_sha256"] or sha256(split_path) != manifest["split_sha256"]:
        raise ValueError("Raw source or frozen split changed since export")
    if sha256(output / "candidates.jsonl") != manifest["candidates_sha256"]:
        raise ValueError("Candidate data was edited; only fill the spot-check decisions")
    candidates = list(map(json.loads, (output / "candidates.jsonl").read_text().splitlines()))
    for row in candidates:
        meta = row["metadata"]
        if (sha256(resolve_path(meta["generation_output"])) != meta["generation_sha256"]
                or sha256(resolve_path(meta["extraction_output"])) != meta["extraction_sha256"]):
            raise ValueError("Saved generation or extraction changed since export")
    text = (output / "spot_check.md").read_text(encoding="utf-8")
    matches = list(re.finditer(r"^## 候选 (synthetic-\d+-[0-9a-f]{16})$", text, re.MULTILINE))
    if [m.group(1) for m in matches] != manifest["spot_check_ids"]:
        raise ValueError("Spot-check IDs or order differ from the exported sample")
    by_id = {row["id"]: row for row in candidates}
    decisions = {}
    for i, match in enumerate(matches):
        identity = match.group(1)
        section = text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        row, source = by_id[identity], manifest["originals"][identity]
        for material in (row["policy"], source["conversation"], source["candidate_response"],
                         row["conversation"], row["candidate_response"]):
            if block(material) not in section:
                raise ValueError(f"Spot-check sample content was edited: {identity}")
        fields, in_code = {}, False
        for line in section.splitlines():
            if line.startswith("````"):
                in_code = not in_code
            elif not in_code:
                for prefix, key in (("结论（通过／剔除／待定）：", "verdict"), ("备注：", "note")):
                    if line.startswith(prefix):
                        if key in fields:
                            raise ValueError(f"Repeated spot-check field: {identity}")
                        fields[key] = line[len(prefix):].strip()
        if fields.get("verdict") not in {"通过", "剔除", "待定"}:
            raise ValueError(f"Fill the conclusion for this spot check: {identity}")
        if fields["verdict"] != "通过" and fields.get("note", "") in {"", "无", "待填写"}:
            raise ValueError(f"Write a brief reason for removing or deferring: {identity}")
        decisions[identity] = {**fields, "reviewer": reviewer}
    kept, removed = [], []
    review_hash = sha256(output / "spot_check.md")
    batch_status = "completed_with_issues" if any(d["verdict"] != "通过" for d in decisions.values()) else "completed"
    for row in candidates:
        decision = decisions.get(row["id"])
        if decision and decision["verdict"] != "通过":
            removed.append(row["id"])
            continue
        row["metadata"].update(batch_spot_check=batch_status, spot_check_sha256=review_hash,
                              batch_spot_check_count=len(decisions), batch_spot_check_reviewer=reviewer,
                              review_status="spot_checked" if decision else "not_individually_reviewed")
        if decision:
            row["metadata"]["spot_check"] = decision
        kept.append(row)
    write_jsonl(output / "training_variants.jsonl", kept)
    write_json(output / "spot_check_results.json", {"decisions": decisions, "removed_ids": removed,
                                                    "spot_check_sha256": review_hash, "reviewer": reviewer})
    (output / "spot_check_snapshot.md").write_text(text, encoding="utf-8")
    summary.update(exported_count=len(kept), removed_after_spot_check_count=len(removed),
                   individually_checked_exported_count=sum(r["id"] in decisions for r in kept),
                   not_individually_reviewed_exported_count=sum(r["id"] not in decisions for r in kept),
                   exported_label_counts=dict(collections.Counter(r["label"] for r in kept)),
                   spot_check_reviewer=reviewer, status=f"{batch_status}; experimental training variants exported; no training run")
    write_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("exported_count", "removed_after_spot_check_count", "not_individually_reviewed_exported_count")}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path, help="Completed batch, expansion or bulk directories")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spot-check-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--apply-spot-check", action="store_true")
    parser.add_argument("--reviewer", choices=("human", "assistant"), default="human")
    args = parser.parse_args()
    try:
        if args.apply_spot_check:
            if args.run_dirs:
                raise ValueError("Use only --output-dir when applying a completed spot check")
            apply_spot_check(resolve_path(args.output_dir), args.reviewer)
        else:
            if not args.run_dirs or args.spot_check_count < 1:
                raise ValueError("Provide a completed run and a positive spot-check count")
            prepare([resolve_path(p) for p in args.run_dirs], resolve_path(args.output_dir),
                    args.spot_check_count, args.seed, resolve_path(args.exclusions))
    except (ValueError, KeyError, IndexError, FileNotFoundError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
