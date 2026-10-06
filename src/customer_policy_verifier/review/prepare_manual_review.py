"""Merge saved batch variants into one source-grouped human review document; no API calls."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT, resolve_path
from customer_policy_verifier.generation.pilot import check_extraction, check_variant
from customer_policy_verifier.data.prepare_data import RAW, SPLIT


def read_json(path: Path) -> dict:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def block(value: str | list[str], language: str = "text") -> str:
    content = "\n".join(value) if isinstance(value, list) else value
    return f"````{language}\n{content}\n````\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_run", type=Path)
    parser.add_argument("expansion_run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    batch, expansion, output = (resolve_path(args.batch_run), resolve_path(args.expansion_run),
                                resolve_path(args.output_dir))
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output directory is not empty; refusing to overwrite human review work")
    selection, summary = read_json(batch / "selection.json"), read_json(batch / "summary.json")
    expand_selection, expand_summary = (read_json(expansion / "selection.json"),
                                       read_json(expansion / "summary.json"))
    split = read_json(SPLIT)
    if hashlib.sha256(resolve_path(RAW).read_bytes()).hexdigest() != selection["source_sha256"] or (
            selection["source_sha256"] != split["source_sha256"]):
        raise SystemExit("Source file or frozen split differs from the batch")
    if resolve_path(expand_selection["base_run"]) != batch:
        raise SystemExit("Expansion references another batch")
    if summary["completed_count"] != len(selection["samples"]) or (
            len(summary["results"]) != len(selection["samples"])):
        raise SystemExit("Batch is incomplete")
    if expand_summary["completed_generations"] != expand_summary["planned_generations"]:
        raise SystemExit("Expansion is incomplete")
    scenes, excluded, fingerprints = {}, [], {}

    def register(sample: dict, extracted: dict, saved: dict, slot: int, path: Path) -> None:
        if split["group_assignment"][str(sample["orig_conv_id"])] != "train":
            raise ValueError("Review sample is outside the training split")
        if extracted["api"]["schema_issues"] or check_extraction(sample, extracted["output"]) or (
                not extracted["output"]["can_expand"]):
            raise ValueError("Source extraction does not pass current checks")
        variant = saved["output"]
        issues = saved["api"]["schema_issues"] or check_variant(sample, variant)
        identity = f"row-{sample['source_row_index']}-v{slot}"
        if issues:
            excluded.append({"variant_id": identity, "reasons": issues})
            return
        key = (sample["source_row_index"], sample["expected_label"])
        canonical = json.dumps({
            "conversation": [" ".join(turn.split()) for turn in variant["conversation"]],
            "candidate_response": " ".join(variant["candidate_response"].split()),
        }, sort_keys=True)
        if (key, canonical) in fingerprints:
            excluded.append({"variant_id": identity, "reasons": ["same-source duplicate"]})
            return
        fingerprints[key, canonical] = identity
        scene = scenes.setdefault(key, {"sample": sample, "extraction": extracted["output"],
                                        "variants": []})
        if scene["sample"] != sample or scene["extraction"] != extracted["output"]:
            raise ValueError("Same source has different source content or extraction")
        scene["variants"].append({
            "variant_id": identity, "variant_number": slot, "output_file": str(path),
            "output_sha256": hashlib.sha256(resolve_path(path).read_bytes()).hexdigest(),
            "conversation": variant["conversation"],
            "candidate_response": variant["candidate_response"],
        })

    base_scenes = {}
    for number, (sample, result) in enumerate(zip(selection["samples"], summary["results"])):
        if sample["source_row_index"] != result["source_row_index"]:
            raise SystemExit("Batch selection and results differ")
        extraction_path = batch / f"sample-{number}-extract.json"
        if extraction_path.exists():
            base_scenes[sample["source_row_index"]] = (sample, read_json(extraction_path))
        if result["status"] in {"candidate_for_human_review", "rejected_review", "rejected_review_format"}:
            path = batch / f"sample-{number}-generate.json"
            register(sample, base_scenes[sample["source_row_index"]][1], read_json(path), 1, path)
    for result in expand_summary["results"]:
        if result["status"] != "candidate_for_human_review":
            continue
        number, slot = result["source_number"], result["variant_number"]
        source = expand_selection["sources"][number]
        sample, extracted = base_scenes[source["sample"]["source_row_index"]]
        if source["sample"] != sample or source["scene"] != extracted["output"]:
            raise SystemExit("Expansion source differs from the original batch")
        path = expansion / f"sample-{number}-generate-{slot}.json"
        register(sample, extracted, read_json(path), slot, path)
    ordered = [scene for _, scene in sorted(scenes.items())]
    variant_count = sum(len(scene["variants"]) for scene in ordered)
    lines = ["# 人工最终复核清单", "", "日期：2026-10-02。", "",
             f"本文件合并首轮与追加生成的候选，共 **{len(ordered)} 个源场景、{variant_count} 条变体**。所有人工结论均待填写。", "",
             "请先查看 docs/guides/manual-review-guide.md。每个场景先复核原标签和提取，再审下面的变体。", "",
             "待审核回复单独列出，尚未成为历史事实；违规样本中的错误主张也应被保留。只判断当前回复，不要求一次完成整段业务流程。", "",
             "填写字段时把“待填写”替换为选定结论，并写一句原文依据；不需要改写样本内容。", "",
             "## 场景索引", "", "| 源行 | 意图 | 预期标签 | 候选编号 |", "| --- | --- | --- | --- |"]
    for scene in ordered:
        sample = scene["sample"]
        ids = ", ".join(v["variant_id"] for v in sorted(scene["variants"], key=lambda v: v["variant_number"]))
        lines.append(f"| {sample['source_row_index']} | {sample['intent']} | {sample['expected_label']} | {ids} |")
    lines += [""]
    for scene in ordered:
        sample = scene["sample"]
        lines += [f"## 源行 {sample['source_row_index']} · {sample['intent']}", "",
                  f"原始对话组：`{sample['orig_conv_id']}`；分区：`train`；预期标签：**{sample['expected_label']}**。", "",
                  "### 源场景复核结论", "",
                  "源标签有依据（是／否／无法确定）：待填写", "",
                  "场景提取准确（是／否／无法确定）：待填写", "",
                  "源场景结论（通过／拒绝／待定）：待填写", "",
                  "源场景依据或疑问：待填写", "",
                  "### 规则原文", "", block(sample["policy"]),
                  "### 原始对话历史", "", block(sample["conversation"]),
                  "### 原始待审核客服回复", "", block(sample["candidate_response"]),
                  "### 场景提取（也是待核对材料）", "",
                  block(json.dumps(scene["extraction"], ensure_ascii=False, indent=2), "json")]
        for variant in sorted(scene["variants"], key=lambda v: v["variant_number"]):
            lines += [f"### 候选 {variant['variant_id']}", "",
                      f"来源记录：`{resolve_path(variant['output_file']).relative_to(ROOT)}`。", "",
                      "变体对话历史：", "", block(variant["conversation"]),
                      "变体待审核客服回复：", "", block(variant["candidate_response"]),
                      "保留性（是／否／无法确定）：待填写", "",
                      "新回复实际标签（符合／违反／无法确定）：待填写", "",
                      "最终结论（通过／拒绝／待定）：待填写", "",
                      "依据或疑问：待填写", ""]
    output.mkdir(parents=True, exist_ok=True)
    (output / "review.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {"batch_run": str(batch), "expansion_run": str(expansion),
                "scene_count": len(ordered), "variant_count": variant_count,
                "label_counts": dict(collections.Counter(
                    scene["sample"]["expected_label"] for scene in ordered for _ in scene["variants"])),
                "excluded_by_current_checks": excluded, "scenes": ordered}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"scene_count": len(ordered), "variant_count": variant_count,
                      "label_counts": manifest["label_counts"], "excluded": excluded,
                      "review_file": str(output / "review.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
