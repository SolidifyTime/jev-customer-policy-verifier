"""Reuse accepted scene extractions to request several distinct variants per source."""

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT, resolve_path
from customer_policy_verifier.generation.batch_experiment import existing_usage, stage_call
from customer_policy_verifier.generation.pilot import (
    MODEL, PROMPT_VERSION, check_extraction, check_variant,
    generation_request, load_key, save_json,
)
from customer_policy_verifier.data.prepare_data import RAW, SPLIT


MULTI_PROMPT_VERSION = "multi-v3"


def load_sources(base_run: Path) -> list[dict]:
    base_run = resolve_path(base_run)
    selection = json.loads((base_run / "selection.json").read_text(encoding="utf-8"))
    summary = json.loads((base_run / "summary.json").read_text(encoding="utf-8"))
    samples = selection["samples"]
    if summary["completed_count"] != len(samples) or len(summary["results"]) != len(samples):
        raise ValueError("Base run is incomplete")
    if selection["model"] != MODEL or selection["prompt_version"] not in {
        "pilot-v6", "pilot-v7", "pilot-v8", "pilot-v9", "pilot-v10", "pilot-v11", PROMPT_VERSION
    }:
        raise ValueError("Base run model or prompt version differs from this expansion script")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if source_hash != split["source_sha256"] or source_hash != selection["source_sha256"]:
        raise ValueError("Raw source or frozen split differs from the base run")
    sources = []
    for number, (sample, result) in enumerate(zip(samples, summary["results"])):
        if sample["source_row_index"] != result["source_row_index"]:
            raise ValueError("Base selection and results are out of order")
        if split["group_assignment"][str(sample["orig_conv_id"])] != "train":
            raise ValueError("A base sample is outside the training split")
        extraction_file = base_run / f"sample-{number}-extract.json"
        generation_file = base_run / f"sample-{number}-generate.json"
        if not extraction_file.exists() or not generation_file.exists():
            continue
        extracted = json.loads(extraction_file.read_text(encoding="utf-8"))
        scene = extracted["output"]
        if extracted["api"]["schema_issues"] or check_extraction(sample, scene) or not scene["can_expand"]:
            continue
        sources.append({"base_number": number, "sample": sample, "scene": scene})
    return sources


def multi_generation_request(sample: dict, scene: dict, prior: list[dict], slot: int):
    request, schema = generation_request(sample, scene)
    request[0]["content"] += (
        " This is an additional wording variant of the same original scene. Choose a different "
        "sentence structure or phrasing from the original and prior variants while preserving "
        "the same facts and response behavior. Prior variants are ONLY examples of wording to "
        "avoid; they may contain mistakes. Never inherit their facts, conditions or actions. "
        "Use the ORIGINAL utterances to resolve any difference. Do not copy variant_number "
        "or prior_variants_to_avoid into the output."
    )
    payload = json.loads(request[1]["content"])
    payload["variant_number"] = slot
    payload["prior_variants_to_avoid"] = prior
    request[1]["content"] = json.dumps(payload, ensure_ascii=False)
    return request, schema


def variant_content(output: dict) -> dict:
    return {"conversation": output.get("conversation"),
            "candidate_response": output.get("candidate_response")}


def evaluate_variant(sample: dict, variant: dict, meta: dict,
                     prior: list[dict]) -> tuple[str, list[str]]:
    problems = meta["schema_issues"] or check_variant(sample, variant)
    if problems:
        return "rejected_format", problems
    if variant_content(variant) in prior:
        return "rejected_duplicate", ["exact prior variant"]
    return "candidate_for_human_review", []


def save_summary(run_dir: Path, selection: dict, results: list[dict], budget: dict) -> None:
    save_json(run_dir / "summary.json", {
        "base_run": selection["base_run"], "model": MODEL,
        "prompt_version": selection["prompt_version"],
        "scene_count": len(selection["sources"]),
        "planned_generations": len(selection["sources"]) * selection["extra_per_scene"],
        "completed_generations": sum(item["status"] != "interrupted" for item in results),
        "api_calls": budget["calls"], "reported_cost_usd": round(budget["cost"], 9),
        "results": results,
    })


def save_review_queue(run_dir: Path, selection: dict, results: list[dict]) -> None:
    sources = selection["sources"]
    lines = ["# 多次生成的待复核变体", "",
             "仅通过程序检查和精确去重；未完成语义复核，未加入训练集。", ""]
    for result in results:
        if result["status"] != "candidate_for_human_review":
            continue
        number, slot = result["source_number"], result["variant_number"]
        source = sources[number]
        sample = source["sample"]
        output = json.loads((run_dir / f"sample-{number}-generate-{slot}.json").read_text())['output']
        lines += [
            f"## 源行 {sample['source_row_index']} · 变体 {slot} · {sample['expected_label']}", "",
            f"原始对话组：`{sample['orig_conv_id']}`；意图：`{sample['intent']}`。", "",
            "### 规则", "", "````text", sample["policy"], "````", "",
            "### 原始对话与回复", "", "````text", *sample["conversation"],
            sample["candidate_response"], "````", "",
            "### 新对话与回复", "", "````text", *output["conversation"],
            output["candidate_response"], "````", "",
            "复核：事实／未知状态／条件关系／回复主张／标签／新增信息量，待填写。", "",
        ]
    (run_dir / "review_queue.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_run", type=Path, help="Completed batch with saved extraction outputs")
    parser.add_argument("--extra-per-scene", type=int, default=2)
    parser.add_argument("--source-rows", nargs="+", type=int,
                        help="Restrict expansion to these source row indexes for a small retest")
    parser.add_argument("--max-cost-usd", type=float, default=0.01)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--recheck", action="store_true",
                        help="Reapply current program checks to saved outputs without API calls")
    args = parser.parse_args()
    if not 1 <= args.extra_per_scene <= 4 or args.max_cost_usd <= 0:
        raise SystemExit("Use 1-4 extra variants per scene and a positive cost limit")
    if args.recheck and not args.resume:
        raise SystemExit("--recheck requires --resume")
    base_run = resolve_path(args.base_run).resolve()
    sources = load_sources(base_run)
    if args.source_rows is not None:
        requested = set(args.source_rows)
        available = {source["sample"]["source_row_index"] for source in sources}
        if len(requested) != len(args.source_rows) or not requested <= available:
            raise SystemExit("--source-rows must contain unique, eligible source row indexes")
        sources = [source for source in sources if source["sample"]["source_row_index"] in requested]
    base_prompt_version = json.loads((base_run / "selection.json").read_text(encoding="utf-8"))["prompt_version"]
    if args.resume:
        run_dir = resolve_path(args.resume).resolve()
        selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
        if (resolve_path(selection["base_run"]).resolve() != base_run
                or selection["extra_per_scene"] != args.extra_per_scene
                or selection["prompt_version"] not in (
                    {"multi-v1", "multi-v2", MULTI_PROMPT_VERSION} if args.recheck else {MULTI_PROMPT_VERSION})
                or selection["sources"] != sources):
            raise SystemExit("Base run, scene selection, variant count, or prompt differs from saved run")
        summary_file = run_dir / "summary.json"
        results = (json.loads(summary_file.read_text(encoding="utf-8"))["results"]
                   if summary_file.exists() else [])
        results = [item for item in results if item["status"] != "interrupted"]
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = ROOT / "runs/generation" / f"expand-{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        selection = {"base_run": str(base_run), "model": MODEL,
                     "base_prompt_version": base_prompt_version,
                     "prompt_version": MULTI_PROMPT_VERSION,
                     "requested_source_rows": args.source_rows,
                     "extra_per_scene": args.extra_per_scene, "sources": sources}
        save_json(run_dir / "selection.json", selection)
        results = []
    print(f"Reusing {len(sources)} scenes for {args.extra_per_scene} extra variants each. "
          f"Run: {run_dir}", flush=True)
    if args.recheck:
        if not args.resume:
            raise SystemExit("--recheck requires --resume")
        checked = []
        for number, source in enumerate(sources):
            sample = source["sample"]
            first = json.loads((base_run / f"sample-{source['base_number']}-generate.json").read_text())['output']
            prior = [variant_content(first)]
            for slot in range(2, args.extra_per_scene + 2):
                path = run_dir / f"sample-{number}-generate-{slot}.json"
                if not path.exists():
                    continue
                saved = json.loads(path.read_text(encoding="utf-8"))
                status, reasons = evaluate_variant(sample, saved["output"], saved["api"], prior)
                item = {"source_number": number, "source_row_index": sample["source_row_index"],
                        "variant_number": slot, "expected_label": sample["expected_label"],
                        "status": status}
                if reasons:
                    item["reasons"] = reasons
                checked.append(item)
                prior.append(variant_content(saved["output"]))
        calls, cost = existing_usage(run_dir)
        budget = {"calls": calls, "cost": cost}
        save_summary(run_dir, selection, checked, budget)
        save_review_queue(run_dir, selection, checked)
        print(f"Rechecked {len(checked)} saved generations: "
              f"{dict(collections.Counter(item['status'] for item in checked))}. No API calls.")
        return
    if not args.execute:
        print("Dry run only; pass --execute to call OpenRouter.")
        return
    key = load_key()
    calls, cost = existing_usage(run_dir)
    budget = {"calls": calls, "cost": cost,
              "max_calls": len(sources) * args.extra_per_scene,
              "max_cost_usd": args.max_cost_usd}
    completed = {(item["source_number"], item["variant_number"]) for item in results}
    for number, source in enumerate(sources):
        sample, scene = source["sample"], source["scene"]
        first = json.loads((base_run / f"sample-{source['base_number']}-generate.json").read_text())['output']
        prior = [variant_content(first)]
        for slot in range(2, args.extra_per_scene + 2):
            if (number, slot) in completed:
                prior.append(variant_content(json.loads(
                    (run_dir / f"sample-{number}-generate-{slot}.json").read_text())['output']))
                continue
            record = {"source_number": number, "source_row_index": sample["source_row_index"],
                      "variant_number": slot, "expected_label": sample["expected_label"]}
            try:
                variant, meta = stage_call(
                    run_dir, number, f"generate-{slot}", sample,
                    multi_generation_request(sample, scene, prior, slot), MODEL, key, budget)
                status, reasons = evaluate_variant(sample, variant, meta, prior)
                record.update(status=status)
                if reasons:
                    record["reasons"] = reasons
            except Exception as exc:
                record.update(status="interrupted", reasons=[str(exc)])
                results.append(record)
                save_summary(run_dir, selection, results, budget)
                print(f"[{number + 1}/{len(sources)} variant {slot}] interrupted: {exc}", flush=True)
                raise
            results.append(record)
            completed.add((number, slot))
            prior.append(variant_content(variant))
            save_summary(run_dir, selection, results, budget)
            print(f"[{number + 1}/{len(sources)} variant {slot}] {record['status']} "
                  f"({budget['calls']} calls, ${budget['cost']:.6f})", flush=True)
    save_review_queue(run_dir, selection, results)
    counts = collections.Counter(item["status"] for item in results)
    print(f"Completed {len(results)} generations: {dict(counts)}. "
          f"Reported cost ${budget['cost']:.6f}. Review queue: {run_dir / 'review_queue.md'}")


if __name__ == "__main__":
    main()
