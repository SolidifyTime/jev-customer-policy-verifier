"""Run a resumable, bounded Flash experiment on up to 50 training sources."""

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT, resolve_path
from customer_policy_verifier.generation.pilot import (
    MODEL, PROMPT_VERSION, call_model, check_extraction, check_variant,
    extraction_request, generation_request, judgment_request, load_key,
    preservation_request, save_json, wire_request, TURN_PREFIX,
)
from customer_policy_verifier.data.prepare_data import RAW, SPLIT, load_rows


def select_sources(rows: list[dict], split: dict, count: int) -> list[dict]:
    """Alternate labels and prefer new intents and conversation groups."""
    if count < 2 or count > 50 or count % 2:
        raise ValueError("--sources must be an even number from 2 to 50")
    choices = {"compliant": [], "violating": []}
    for index, row in enumerate(rows):
        if split["group_assignment"][str(row["orig_conv_id"])] != "train":
            continue
        for label, field in (("compliant", "orig_gt_response"),
                             ("violating", "manipulated_gt_response")):
            if not TURN_PREFIX.match(row[field]):
                continue
            digest = hashlib.sha256(f"{split['seed']}:{index}:{label}".encode()).hexdigest()
            choices[label].append((digest, index, row, field))
    selected = []
    used_groups = set()
    used_intents = set()
    for slot in range(count):
        label = "compliant" if slot % 2 == 0 else "violating"
        eligible = [item for item in choices[label]
                    if item[2]["orig_conv_id"] not in used_groups]
        fresh_intents = [item for item in eligible if item[2]["intent"] not in used_intents]
        if not eligible:
            raise ValueError(f"Not enough eligible {label} training groups")
        _, index, row, field = min(fresh_intents or eligible)
        used_groups.add(row["orig_conv_id"])
        used_intents.add(row["intent"])
        selected.append({
            "source_row_index": index, "orig_conv_id": row["orig_conv_id"],
            "intent": row["intent"], "policy": row["instruction"],
            "conversation": row["conversation"], "candidate_response": row[field],
            "expected_label": label,
        })
    return selected


def existing_usage(run_dir: Path) -> tuple[int, float]:
    run_dir = resolve_path(run_dir)
    count = 0
    cost = 0.0
    for path in run_dir.glob("sample-*-*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or "api" not in record:
            continue
        count += 1
        reported = record["api"].get("usage", {}).get("cost")
        if not isinstance(reported, (int, float)):
            raise ValueError(f"Missing API cost in {path}")
        cost += reported
    return count, cost


def stage_call(run_dir: Path, number: int, stage: str, sample: dict,
               request: tuple[list[dict], dict], model: str, key: str,
               budget: dict) -> tuple[dict, dict]:
    run_dir = resolve_path(run_dir)
    path = run_dir / f"sample-{number}-{stage}.json"
    request_messages, schema = request
    request_messages, response_format = wire_request(model, request_messages, schema)
    request_file = run_dir / f"sample-{number}-{stage}-request.json"
    request_data = {"messages": request_messages, "response_format": response_format}
    if request_file.exists():
        if json.loads(request_file.read_text(encoding="utf-8")) != request_data:
            raise ValueError(f"Prompt changed during resumed run: {request_file}")
    else:
        save_json(request_file, request_data)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["output"], cached["api"]
    if budget["calls"] >= budget["max_calls"]:
        raise RuntimeError("Experiment API call limit reached")
    if budget["cost"] >= budget["max_cost_usd"]:
        raise RuntimeError("Experiment reported-cost limit reached")
    output, api = call_model(key, model, request_messages, response_format, schema)
    save_json(path, {"output": output, "api": api})
    budget["calls"] += 1
    reported = api.get("usage", {}).get("cost")
    if not isinstance(reported, (int, float)):
        raise RuntimeError(f"OpenRouter did not report cost for {path}")
    budget["cost"] += reported
    return output, api


def save_summary(run_dir: Path, model: str, review_mode: str, selected: list[dict],
                 results: list[dict], budget: dict) -> None:
    save_json(run_dir / "summary.json", {
        "model": model, "prompt_version": PROMPT_VERSION, "review_mode": review_mode,
        "source_count": len(selected), "completed_count": sum(
            item["status"] != "interrupted" for item in results),
        "api_calls": budget["calls"], "reported_cost_usd": round(budget["cost"], 9),
        "results": results,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=50)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--review-mode", choices=("manual", "model"), default="manual",
                        help="Manual queue after program checks (default), or two model review calls")
    parser.add_argument("--max-cost-usd", type=float, default=0.05)
    parser.add_argument("--resume", type=Path, help="Resume a run without repeating saved API calls")
    args = parser.parse_args()
    if args.max_cost_usd <= 0:
        raise SystemExit("--max-cost-usd must be positive")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    if hashlib.sha256(RAW.read_bytes()).hexdigest() != split["source_sha256"]:
        raise SystemExit("Raw data hash differs from the frozen split")
    if args.resume:
        run_dir = resolve_path(args.resume).resolve()
        selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
        if (selection["model"] != args.model or selection["prompt_version"] != PROMPT_VERSION
                or selection.get("review_mode", "model") != args.review_mode
                or selection["source_sha256"] != split["source_sha256"]):
            raise SystemExit("Model, prompt, review mode, or source data differ from the saved run")
        selected = selection["samples"]
        summary_file = run_dir / "summary.json"
        results = (json.loads(summary_file.read_text(encoding="utf-8"))["results"]
                   if summary_file.exists() else [])
        results = [item for item in results if item["status"] != "interrupted"]
    else:
        selected = select_sources(load_rows(RAW), split, args.sources)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = ROOT / "runs/generation" / f"batch-{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        save_json(run_dir / "selection.json", {
            "model": args.model, "prompt_version": PROMPT_VERSION,
            "review_mode": args.review_mode,
            "source_sha256": split["source_sha256"],
            "selection_rule": "alternating labels; new intents first; unique original conversations; seeded hash tie-break",
            "samples": selected,
        })
        results = []
    label_counts = collections.Counter(item["expected_label"] for item in selected)
    group_count = len({item["orig_conv_id"] for item in selected})
    intent_count = len({item["intent"] for item in selected})
    print(f"Selected {len(selected)} sources: {dict(label_counts)}, {group_count} groups, "
          f"{intent_count} intents. Review: {args.review_mode}. Run: {run_dir}", flush=True)
    if not args.execute:
        print("Dry run only; pass --execute to call OpenRouter.")
        return
    key = load_key()
    calls, cost = existing_usage(run_dir)
    budget = {"calls": calls, "cost": cost,
              "max_calls": len(selected) * (2 if args.review_mode == "manual" else 4),
              "max_cost_usd": args.max_cost_usd}
    completed = {item["source_row_index"] for item in results}
    for number, sample in enumerate(selected):
        if sample["source_row_index"] in completed:
            continue
        record = {"source_row_index": sample["source_row_index"],
                  "orig_conv_id": sample["orig_conv_id"], "intent": sample["intent"],
                  "expected_label": sample["expected_label"]}
        try:
            scene, meta = stage_call(run_dir, number, "extract", sample,
                                     extraction_request(sample), args.model, key, budget)
            problems = meta["schema_issues"] or check_extraction(sample, scene)
            if problems or not scene.get("can_expand"):
                issues = scene.get("issues", [])
                if not isinstance(issues, list):
                    issues = []
                record.update(status="rejected_extraction", reasons=problems + issues)
            else:
                variant, meta = stage_call(run_dir, number, "generate", sample,
                                           generation_request(sample, scene), args.model, key, budget)
                problems = meta["schema_issues"] or check_variant(sample, variant)
                if problems:
                    record.update(status="rejected_format", reasons=problems)
                elif args.review_mode == "manual":
                    record.update(status="candidate_for_human_review", variant={
                        "policy": sample["policy"], "conversation": variant["conversation"],
                        "candidate_response": variant["candidate_response"],
                        "label": sample["expected_label"]})
                else:
                    preservation, meta = stage_call(
                        run_dir, number, "preserve", sample,
                        preservation_request(sample, scene, variant), args.model, key, budget)
                    if meta["schema_issues"]:
                        record.update(status="rejected_review_format", reasons=meta["schema_issues"])
                    else:
                        judgment, meta = stage_call(
                            run_dir, number, "judge", sample,
                            judgment_request(sample, variant), args.model, key, budget)
                        if meta["schema_issues"]:
                            record.update(status="rejected_review_format", reasons=meta["schema_issues"])
                        elif (preservation.get("verdict") == "pass"
                              and judgment.get("verdict") == sample["expected_label"]):
                            record.update(status="candidate_for_human_review", variant={
                                "policy": sample["policy"], "conversation": variant["conversation"],
                                "candidate_response": variant["candidate_response"],
                                "label": sample["expected_label"]})
                        else:
                            record.update(status="rejected_review", preservation=preservation,
                                          judgment=judgment)
        except Exception as exc:
            record.update(status="interrupted", reasons=[str(exc)])
            results.append(record)
            save_summary(run_dir, args.model, args.review_mode, selected, results, budget)
            print(f"[{number + 1}/{len(selected)}] interrupted: {exc}", flush=True)
            raise
        results.append(record)
        completed.add(sample["source_row_index"])
        save_summary(run_dir, args.model, args.review_mode, selected, results, budget)
        print(f"[{number + 1}/{len(selected)}] {sample['expected_label']} "
              f"{sample['intent']}: {record['status']} "
              f"({budget['calls']} calls, ${budget['cost']:.6f})", flush=True)
    print(f"Completed {len(selected)} sources; summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
