"""Validate a completed human review and export only accepted training variants; no API calls."""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT, resolve_path
from customer_policy_verifier.data.prepare_data import RAW, SPLIT, load_rows
from customer_policy_verifier.review.prepare_manual_review import block


ANSWERS = {"是", "否", "无法确定"}
VERDICTS = {"通过", "拒绝", "待定"}
LABELS = {"符合": "compliant", "违反": "violating", "无法确定": "uncertain"}
SOURCE_FIELDS = {
    "源标签有依据（是／否／无法确定）": "label_supported",
    "场景提取准确（是／否／无法确定）": "extraction_accurate",
    "源场景结论（通过／拒绝／待定）": "verdict",
    "源场景依据或疑问": "reason",
}
VARIANT_FIELDS = {
    "保留性（是／否／无法确定）": "preserved",
    "新回复实际标签（符合／违反／无法确定）": "actual_label",
    "最终结论（通过／拒绝／待定）": "verdict",
    "依据或疑问": "reason",
}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def split_sections(text: str, pattern: str) -> dict[str, str]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    sections = {}
    for i, match in enumerate(matches):
        key = match.group(1)
        if key in sections:
            raise ValueError(f"Duplicate review section: {key}")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[key] = text[match.end():end]
    return sections


def read_fields(section: str, specs: dict[str, str]) -> dict[str, str]:
    values = {}
    inside_code = False
    for line in section.splitlines():
        if line.startswith("````"):
            inside_code = not inside_code
            continue
        if inside_code:
            continue
        for prefix, field in specs.items():
            if line.startswith(prefix + "："):
                if field in values:
                    raise ValueError(f"Repeated review field: {prefix}")
                value = line[len(prefix) + 1:].strip()
                if not value or "待填写" in value:
                    raise ValueError(f"Unfilled review field: {prefix}")
                values[field] = value
    if set(values) != set(specs.values()):
        raise ValueError(f"Missing review fields: {set(specs.values()) - set(values)}")
    if values["verdict"] not in VERDICTS:
        raise ValueError(f"Unexpected verdict: {values['verdict']}")
    for field in ("label_supported", "extraction_accurate", "preserved"):
        if field in values and values[field] not in ANSWERS:
            raise ValueError(f"Unexpected answer for {field}: {values[field]}")
    if "actual_label" in values:
        if values["actual_label"] not in LABELS:
            raise ValueError(f"Unexpected actual label: {values['actual_label']}")
        values["actual_label"] = LABELS[values["actual_label"]]
    return values


def parse_review(text: str, manifest: dict) -> list[dict]:
    source_sections = split_sections(text, r"^## 源行 (\d+) · [^\n]+$")
    expected = {str(scene["sample"]["source_row_index"]) for scene in manifest["scenes"]}
    if set(source_sections) != expected:
        raise ValueError("Review source IDs differ from the prepared manifest")
    decisions = []
    for scene in manifest["scenes"]:
        sample = scene["sample"]
        section = source_sections[str(sample["source_row_index"])]
        source_text = section.split("### 候选 ", 1)[0]
        source_decision = read_fields(source_text, SOURCE_FIELDS)
        if source_decision["verdict"] == "通过" and (
                source_decision["label_supported"] != "是"
                or source_decision["extraction_accurate"] != "是"):
            raise ValueError(f"Accepted source has inconsistent answers: {sample['source_row_index']}")
        for original in (block(sample["policy"]), block(sample["conversation"]),
                         block(sample["candidate_response"]),
                         block(json.dumps(scene["extraction"], ensure_ascii=False, indent=2), "json")):
            if original not in source_text:
                raise ValueError(f"Source materials were edited: {sample['source_row_index']}")
        variant_sections = split_sections(section, r"^### 候选 (row-\d+-v\d+)$")
        if set(variant_sections) != {v["variant_id"] for v in scene["variants"]}:
            raise ValueError(f"Variant IDs differ in source {sample['source_row_index']}")
        variants = []
        for variant in scene["variants"]:
            identity = variant["variant_id"]
            content = variant_sections[identity]
            for original in (block(variant["conversation"]), block(variant["candidate_response"])):
                if original not in content:
                    raise ValueError(f"Generated material was edited: {identity}")
            decision = read_fields(content, VARIANT_FIELDS)
            if decision["verdict"] == "通过" and (
                    source_decision["verdict"] != "通过" or decision["preserved"] != "是"
                    or decision["actual_label"] != sample["expected_label"]):
                raise ValueError(f"Accepted variant has inconsistent decisions: {identity}")
            variants.append({"variant_id": identity, **decision})
        decisions.append({"source_row_index": sample["source_row_index"],
                          "expected_label": sample["expected_label"],
                          "source_review": source_decision, "variant_reviews": variants})
    return decisions


def validate_provenance(manifest: dict) -> dict:
    split = json.loads(resolve_path(SPLIT).read_text(encoding="utf-8"))
    if digest(resolve_path(RAW).read_bytes()) != split["source_sha256"]:
        raise ValueError("Raw source differs from the frozen split")
    rows = load_rows(RAW)
    for scene in manifest["scenes"]:
        sample = scene["sample"]
        original = rows[sample["source_row_index"]]
        response_field = {"compliant": "orig_gt_response", "violating": "manipulated_gt_response"}[
            sample["expected_label"]]
        if (sample["policy"] != original["instruction"]
                or sample["conversation"] != original["conversation"]
                or sample["candidate_response"] != original[response_field]
                or sample["orig_conv_id"] != original["orig_conv_id"]
                or sample["intent"] != original["intent"]):
            raise ValueError(f"Review source differs from raw data: {sample['source_row_index']}")
        if split["group_assignment"][str(sample["orig_conv_id"])] != "train":
            raise ValueError("Reviewed source is outside the training split")
        for variant in scene["variants"]:
            saved_bytes = resolve_path(variant["output_file"]).read_bytes()
            if digest(saved_bytes) != variant["output_sha256"]:
                raise ValueError(f"Saved generation changed: {variant['variant_id']}")
            generated = json.loads(saved_bytes)["output"]
            if (generated["conversation"] != variant["conversation"]
                    or generated["candidate_response"] != variant["candidate_response"]):
                raise ValueError(f"Manifest differs from saved generation: {variant['variant_id']}")
    return split


def jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    review_dir, output = resolve_path(args.review_dir), resolve_path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output directory is not empty; refusing to overwrite an imported review")
    review_bytes = (review_dir / "review.md").read_bytes()
    manifest_bytes = (review_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    split = validate_provenance(manifest)
    decisions = parse_review(review_bytes.decode("utf-8"), manifest)
    accepted, exclusions, source_audit = [], [], []
    for scene, decision in zip(manifest["scenes"], decisions):
        sample = scene["sample"]
        source_review = decision["source_review"]
        source_audit.append({
            "source_row_index": sample["source_row_index"], "orig_conv_id": sample["orig_conv_id"],
            "intent": sample["intent"], "split": "train", "expected_label": sample["expected_label"],
            **source_review,
        })
        if source_review["label_supported"] != "是":
            exclusions.append({"source_row_index": sample["source_row_index"],
                               "expected_label": sample["expected_label"],
                               "label_supported": source_review["label_supported"],
                               "reason": source_review["reason"]})
        reviews = {item["variant_id"]: item for item in decision["variant_reviews"]}
        for variant in scene["variants"]:
            reviewed = reviews[variant["variant_id"]]
            if reviewed["verdict"] != "通过":
                continue
            accepted.append({
                "id": variant["variant_id"], "policy": sample["policy"],
                "conversation": variant["conversation"], "candidate_response": variant["candidate_response"],
                "label": sample["expected_label"],
                "metadata": {"source_type": "synthetic", "split": "train",
                             "source_row_index": sample["source_row_index"],
                             "orig_conv_id": sample["orig_conv_id"], "intent": sample["intent"],
                             "variant_number": variant["variant_number"],
                             "generation_output": str(resolve_path(variant["output_file"]).relative_to(ROOT)),
                             "generation_sha256": variant["output_sha256"],
                             "source_sha256": split["source_sha256"],
                             "review_sha256": digest(review_bytes),
                             "source_review": source_review, "variant_review": reviewed},
            })
    source_counts = dict(collections.Counter(d["source_review"]["verdict"] for d in decisions))
    variant_counts = dict(collections.Counter(v["verdict"] for d in decisions for v in d["variant_reviews"]))
    summary = {
        "review_file": str(review_dir / "review.md"), "review_sha256": digest(review_bytes),
        "manifest_sha256": digest(manifest_bytes), "source_sha256": split["source_sha256"],
        "source_count": len(decisions), "source_verdict_counts": source_counts,
        "variant_count": sum(len(d["variant_reviews"]) for d in decisions),
        "variant_verdict_counts": variant_counts, "accepted_count": len(accepted),
        "accepted_label_counts": dict(collections.Counter(item["label"] for item in accepted)),
        "accepted_source_count": len({item["metadata"]["source_row_index"] for item in accepted}),
        "accepted_intent_count": len({item["metadata"]["intent"] for item in accepted}),
        "reviewed_original_candidates_to_exclude_from_baseline": exclusions,
        "status": "reviewed synthetic training candidates; model training not started",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "accepted_variants.jsonl").write_text(jsonl(accepted), encoding="utf-8")
    (output / "source_audit.jsonl").write_text(jsonl(source_audit), encoding="utf-8")
    (output / "baseline_exclusions.jsonl").write_text(jsonl(exclusions), encoding="utf-8")
    (output / "decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "review_snapshot.md").write_bytes(review_bytes)
    print(json.dumps({key: summary[key] for key in (
        "source_verdict_counts", "variant_verdict_counts", "accepted_count",
        "accepted_label_counts", "accepted_source_count", "accepted_intent_count")}, ensure_ascii=False, indent=2))
    print(f"Exported reviewed candidates: {output / 'accepted_variants.jsonl'}")


if __name__ == "__main__":
    main()
