"""Summarize a completed local batch and make a candidate review queue."""

import argparse
import collections
import json
from pathlib import Path

from customer_policy_verifier.common.paths import resolve_path

def block(value: str | list[str]) -> str:
    text = "\n".join(value) if isinstance(value, list) else value
    return "````text\n" + text + "\n````\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--include-model-rejected", action="store_true",
                        help="Queue every variant that passed program checks, regardless of model review")
    args = parser.parse_args()
    run_dir = resolve_path(args.run_dir)
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    samples = selection["samples"]
    results = summary["results"]
    if len(samples) != len(results) or summary["completed_count"] != len(samples):
        raise SystemExit("Batch has not completed every selected source")
    for sample, result in zip(samples, results):
        if sample["source_row_index"] != result["source_row_index"]:
            raise SystemExit("Selection and result order differ")
    statuses = collections.Counter(result["status"] for result in results)
    labels = {label: dict(collections.Counter(result["status"] for result in results
                                              if result["expected_label"] == label))
              for label in ("compliant", "violating")}
    reasons = collections.Counter(reason for result in results
                                  for reason in result.get("reasons", []))
    stages = collections.Counter()
    for path in run_dir.glob("sample-*-*.json"):
        if "api" in json.loads(path.read_text(encoding="utf-8")):
            stages[path.stem.rsplit("-", 1)[-1]] += 1
    review_mode = selection.get("review_mode", "model")
    aggregate = {
        "review_mode": review_mode,
        "source_count": len(samples),
        "original_conversation_groups": len({sample["orig_conv_id"] for sample in samples}),
        "intents": len({sample["intent"] for sample in samples}),
        "status_counts": dict(statuses), "status_by_label": labels,
        "stage_call_counts": dict(stages), "reason_counts": dict(reasons),
        "api_calls": summary["api_calls"],
        "reported_cost_usd": summary["reported_cost_usd"],
    }
    cost_key = ("cost_per_program_check_candidate_usd" if review_mode == "manual"
                else "cost_per_auto_candidate_usd")
    aggregate[cost_key] = (summary["reported_cost_usd"] /
                           statuses["candidate_for_human_review"]
                           if statuses["candidate_for_human_review"] else None)
    (run_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.include_model_rejected:
        statuses_to_queue = {"candidate_for_human_review", "rejected_review",
                             "rejected_review_format"}
        queue_file = run_dir / "manual_review_queue.md"
        description = "列出全部通过提取与变体程序检查的样本；模型审核结论仅供参考，不作为本清单的筛选条件。"
    else:
        statuses_to_queue = {"candidate_for_human_review"}
        queue_file = run_dir / "review_queue.md"
        description = ("仅列通过程序检查的变体，尚未加入训练集。" if review_mode == "manual"
                       else "仅列模型审核通过的变体，尚未加入训练集。")
    lines = ["# 待复核候选", "", description, "", "复核时逐项检查：原标签是否有依据；关键事实、未知状态、条件关系和回复主张是否保留；变体是否有实际增量。", ""]
    for number, (sample, result) in enumerate(zip(samples, results)):
        if result["status"] not in statuses_to_queue:
            continue
        extracted = json.loads((run_dir / f"sample-{number}-extract.json").read_text())['output']
        generated = json.loads((run_dir / f"sample-{number}-generate.json").read_text())['output']
        lines += [
            f"## 源行 {sample['source_row_index']} · {sample['intent']} · {sample['expected_label']}",
            "", f"原始对话组：`{sample['orig_conv_id']}`；批次序号：`{number}`；原运行状态：`{result['status']}`。", "",
            "### 规则原文", "", block(sample["policy"]),
            "### 原始对话与回复", "", block(sample["conversation"]),
            block(sample["candidate_response"]),
            "### 变体对话与回复", "", block(generated["conversation"]),
            block(generated["candidate_response"]),
            "### 未确认信息", "",
            *(f"- {item}" for item in extracted.get("unknowns", [])), "",
        ]
        preserve_file = run_dir / f"sample-{number}-preserve.json"
        judge_file = run_dir / f"sample-{number}-judge.json"
        if preserve_file.exists():
            preservation = json.loads(preserve_file.read_text())['output']
            lines += [f"模型事实保留意见：**{preservation.get('verdict', 'unknown')}**。{preservation.get('reason', '')}", ""]
        if judge_file.exists():
            judgment = json.loads(judge_file.read_text())['output']
            lines += [f"模型独立判断意见：**{judgment.get('verdict', 'unknown')}**。{judgment.get('reason', '')}", ""]
        lines += ["复核结论：待填写。", ""]
    queue_file.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: aggregate[key] for key in (
        "source_count", "review_mode", "status_counts", "stage_call_counts",
        "api_calls", "reported_cost_usd", cost_key)}, ensure_ascii=False, indent=2))
    print(f"Review queue: {queue_file}")


if __name__ == "__main__":
    main()
