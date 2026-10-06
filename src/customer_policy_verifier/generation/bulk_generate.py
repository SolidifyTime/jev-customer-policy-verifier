"""Prepare a frozen OpenRouter batch; paid calls require explicit --execute and budget."""

import argparse
import collections
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT, resolve_path
from customer_policy_verifier.generation import compatible_api
from customer_policy_verifier.generation.expand_scenes import (
    MULTI_PROMPT_VERSION, multi_generation_request, variant_content,
)
from customer_policy_verifier.generation.pilot import (
    PROMPT_VERSION, check_extraction, check_variant, extraction_request,
    generation_request, load_key, schema_issues, wire_request,
)
from customer_policy_verifier.data.prepare_data import load_rows


VERSION = f"bulk-v1/{PROMPT_VERSION}/{MULTI_PROMPT_VERSION}"
DATASET = ROOT / "data/source/task2/manifest.json"
EXCLUSIONS = ROOT / "data/archive/reviews/manual-review-20261002/baseline_exclusions.jsonl"
TERMINAL = {"completed", "rejected_extraction"}


def read_json(path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_directory(run, number):
    return resolve_path(run) / "sources" / f"{number:05d}"


def implementation_hashes():
    return {name: digest(Path(__file__).parent / name) for name in
            ("bulk_generate.py", "compatible_api.py", "pilot.py", "expand_scenes.py")}


def select_sources(rows, split, exclusions):
    selected, skipped = [], []
    for index, row in enumerate(rows):
        if split["group_assignment"][str(row["orig_conv_id"])] != "train":
            continue
        for label, field in (("compliant", "orig_gt_response"),
                             ("violating", "manipulated_gt_response")):
            reason = None
            if (index, label) in exclusions:
                reason = "previously reviewed unsupported or uncertain source label"
            elif not re.match(r"^turn \d+: agent:", row[field]):
                reason = "source response lacks turn N: agent: prefix"
            if reason:
                skipped.append({"source_row_index": index, "expected_label": label, "reason": reason})
                continue
            selected.append({"source_row_index": index, "orig_conv_id": row["orig_conv_id"],
                             "intent": row["intent"], "policy": row["instruction"],
                             "conversation": row["conversation"], "candidate_response": row[field],
                             "expected_label": label})
    # Keep both labels of a row. Deterministic ordering spreads groups across early checkpoints.
    selected.sort(key=lambda sample: hashlib.sha256(
        f"{split['seed']}:{sample['source_row_index']}:{sample['expected_label']}".encode()).hexdigest())
    return selected, skipped


def build_selection(manifest_path, variants):
    manifest_path = resolve_path(manifest_path)
    manifest = read_json(manifest_path)
    raw, split_path = resolve_path(manifest["source_file"]), resolve_path(manifest["split_file"])
    rows, split = load_rows(raw), read_json(split_path)
    if digest(raw) != manifest["source_sha256"] or digest(raw) != split["source_sha256"]:
        raise ValueError("Dataset source changed; recheck the frozen manifest")
    if digest(split_path) != manifest["split_sha256"]:
        raise ValueError("Dataset split changed")
    exclusions = set()
    eligibility = manifest.get("generation_eligibility")
    exclusion_hash = None
    if eligibility:
        exclusion_path = resolve_path(eligibility["exclusions_file"])
        exclusion_hash = digest(exclusion_path)
        if exclusion_hash != eligibility["exclusions_sha256"]:
            raise ValueError("Frozen generation exclusions changed")
        exclusions = {(row["source_row_index"], row["expected_label"])
                      for row in read_json(exclusion_path)["historical_decisions"]}
    elif EXCLUSIONS.exists():
        exclusions = {(row["source_row_index"], row["expected_label"])
                      for row in map(json.loads, EXCLUSIONS.read_text().splitlines())}
        exclusion_hash = digest(EXCLUSIONS)
    samples, skipped = select_sources(rows, split, exclusions)
    if eligibility and len(samples) != eligibility["eligible_candidates"]:
        raise ValueError("Generation selection differs from the dataset eligibility audit")
    config = compatible_api.load_config()
    return {"kind": "bulk", "model": config["model"], "prompt_version": VERSION,
            "source_file": str(raw.resolve()), "source_sha256": digest(raw),
            "split_file": str(split_path.resolve()), "split_sha256": digest(split_path),
            "dataset_manifest": str(manifest_path.resolve()), "dataset_manifest_sha256": digest(manifest_path),
            "exclusions_sha256": exclusion_hash,
            "implementation_sha256": implementation_hashes(), "api_config": config,
            "variants_per_scene": variants, "samples": samples, "skipped_sources": skipped}


def request_body(config, request):
    messages, schema = request
    messages, response_format = wire_request(config["model"], messages, schema, json_mode=True)
    body = {key: config[key] for key in ("model", "temperature", "max_tokens", "reasoning", "provider")}
    body.update(messages=messages, response_format=response_format, stream=False)
    return body, schema


def frozen_plan(selection, concurrency):
    samples = selection["samples"]
    return {"status": "prepared_awaiting_user_confirmation", "model": selection["model"],
            "api_base": selection["api_config"]["api_base"], "source_count": len(samples),
            "source_label_counts": dict(collections.Counter(s["expected_label"] for s in samples)),
            "source_group_count": len({s["orig_conv_id"] for s in samples}),
            "skipped_source_count": len(selection["skipped_sources"]),
            "variants_per_scene": selection["variants_per_scene"],
            "maximum_generation_calls": len(samples) * selection["variants_per_scene"],
            "maximum_logical_calls": len(samples) * (selection["variants_per_scene"] + 1),
            "initial_concurrency": concurrency, "spot_check_count_for_entire_batch": 3,
            "model_review_calls": 0,
            "notes": ["Only training sources are sent to the model.",
                      f"Rejected extractions do not generate; {selection['variants_per_scene']} attempts do not guarantee that many accepted variants.",
                      "Transient transport failures retry at most 4 attempts per logical call.",
                      "Budget is a reported-cost stop threshold; already in-flight requests may exceed it.",
                      "Missing usage.cost pauses execution after saving the response."]}


class Batch:
    def __init__(self, run, selection, concurrency, max_cost, key, call=None):
        self.run, self.selection, self.key = resolve_path(run), selection, key
        self.gate = compatible_api.RequestGate(concurrency)
        self.max_cost = max_cost
        self.call = call or compatible_api.call
        self.lock = threading.Lock()
        self.results = {}
        self.records = {}
        self.reason = None
        for number in range(len(selection["samples"])):
            directory = source_directory(run, number)
            result_path = directory / "result.json"
            if result_path.exists():
                self.results[number] = read_json(result_path)
            for path in [directory / "extract.json", *[p for p in directory.glob("generate-*.json")
                         if re.fullmatch(r"generate-\d+\.json", p.name)]]:
                if path.exists():
                    self.records[str(path)] = read_json(path)["api"]

    def usage(self):
        records = list(self.records.values())
        costs = [record.get("reported_cost_usd") for record in records]
        known = [cost for cost in costs if isinstance(cost, (int, float)) and not isinstance(cost, bool)]
        return {"saved_api_responses": len(records),
                "successful_stage_http_attempts": sum(len(r.get("attempts", [])) for r in records),
                "reported_cost_subtotal_usd": round(sum(known), 9),
                "missing_cost_count": len(costs) - len(known),
                "reported_cost_usd": round(sum(known), 9) if len(known) == len(costs) else None,
                "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) or 0 for r in records),
                "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) or 0 for r in records)}

    def stop(self, reason):
        with self.lock:
            if not self.reason:
                self.reason = reason
            self.gate.stop.set()

    def check_budget(self):
        with self.lock:
            usage = self.usage()
        if usage["missing_cost_count"]:
            self.stop("Missing API cost; saved output retained, reconcile before resuming")
        elif usage["reported_cost_subtotal_usd"] >= self.max_cost:
            self.stop("Reported-cost threshold reached")
        if self.gate.stop.is_set():
            raise InterruptedError(self.reason or "Run interrupted")

    def stage(self, directory, name, request):
        body, schema = request_body(self.selection["api_config"], request)
        request_path, output_path = directory / f"{name}-request.json", directory / f"{name}.json"
        if request_path.exists() and read_json(request_path) != body:
            raise ValueError(f"Cached request differs: {request_path}")
        if output_path.exists():
            saved = read_json(output_path)
            if not request_path.exists() or saved.get("request_sha256") != digest(request_path):
                raise ValueError(f"Cached response request hash differs: {output_path}")
            return saved["output"], saved["api"]
        self.check_budget()
        atomic_json(request_path, body)
        try:
            output, api = self.call(self.selection["api_config"], self.key, body, schema, request_path, self.gate)
        except compatible_api.ApiFailure as exc:
            atomic_json(directory / f"{name}-error.json", {"error": str(exc), "attempts": exc.attempts})
            raise
        # Save every returned response, including invalid JSON or truncated content, before checks.
        atomic_json(output_path, {"output": output, "api": api, "request_sha256": digest(request_path)})
        with self.lock:
            self.records[str(output_path)] = api
        return output, api

    def checkpoint(self, number, result):
        atomic_json(source_directory(self.run, number) / "result.json", result)
        with self.lock:
            self.results[number] = json.loads(json.dumps(result))

    def source(self, number):
        sample = self.selection["samples"][number]
        directory = source_directory(self.run, number)
        result = {"source_number": number, "source_row_index": sample["source_row_index"],
                  "expected_label": sample["expected_label"], "status": "running", "generations": []}
        try:
            scene, api = self.stage(directory, "extract", extraction_request(sample))
            issues = api["schema_issues"] or schema_issues(scene, extraction_request(sample)[1]["json_schema"]["schema"])
            issues = issues or check_extraction(sample, scene)
            if issues or not scene["can_expand"]:
                result.update(status="rejected_extraction", issues=issues or scene["issues"] or ["can_expand=false"])
                self.checkpoint(number, result)
                return
            prior, seen = [], set()
            result["extraction_passed"] = True
            self.checkpoint(number, result)
            for slot in range(1, self.selection["variants_per_scene"] + 1):
                request = (generation_request(sample, scene) if slot == 1 else
                           multi_generation_request(sample, scene, prior[-3:], slot))
                variant, meta = self.stage(directory, f"generate-{slot:02d}", request)
                problems = meta["schema_issues"] or schema_issues(variant, request[1]["json_schema"]["schema"])
                problems = problems or check_variant(sample, variant)
                status = "rejected_format" if problems else "candidate_for_human_review"
                if not problems:
                    content = variant_content(variant)
                    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False)
                    normalized = " ".join(canonical.split())
                    if normalized in seen:
                        status, problems = "rejected_duplicate", ["duplicate variant of this scene"]
                    else:
                        seen.add(normalized)
                        prior.append(content)
                result["generations"].append({"variant_number": slot, "status": status, "issues": problems})
                self.checkpoint(number, result)
            result["status"] = "completed"
        except Exception as exc:
            # Only this stage is retried on resume; earlier saved outputs are reused.
            result.update(status="interrupted", error=str(exc).replace(self.key, "[REDACTED]"))
            self.stop(result["error"])
        self.checkpoint(number, result)

    def summary(self, status):
        with self.lock:
            results = [self.results[n] for n in sorted(self.results)]
            usage = self.usage()
            summary = {"model": self.selection["model"], "prompt_version": self.selection["prompt_version"],
                       "status": status, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "source_count": len(self.selection["samples"]),
                       "completed_count": sum(r["status"] in TERMINAL for r in results),
                       "extraction_passed_count": sum(r.get("extraction_passed", False) for r in results),
                       "completed_generations": sum(len(r["generations"]) for r in results),
                       "generation_status_counts": dict(collections.Counter(
                           g["status"] for r in results for g in r["generations"])),
                       "effective_concurrency": self.gate.limit,
                       "rate_limit_events_this_invocation": self.gate.rate_limit_events,
                       "stop_reason": self.reason, "usage": usage, "results": results}
        atomic_json(self.run / "summary.json", summary)
        return summary

    def execute(self, max_sources=None):
        pending = [n for n in range(len(self.selection["samples"]))
                   if self.results.get(n, {}).get("status") not in TERMINAL]
        if max_sources is not None:
            pending = pending[:max_sources]
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.gate.limit)
        futures = set()
        iterator = iter(pending)
        try:
            self.check_budget()
            for _ in range(self.gate.limit):
                number = next(iterator, None)
                if number is not None:
                    futures.add(executor.submit(self.source, number))
            last_report = 0.0
            while futures:
                done, futures = concurrent.futures.wait(futures, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    future.result()
                    if not self.gate.stop.is_set():
                        number = next(iterator, None)
                        if number is not None:
                            futures.add(executor.submit(self.source, number))
                if done or time.monotonic() - last_report >= 10:
                    summary = self.summary("running")
                    print(json.dumps({key: summary[key] for key in
                                      ("completed_count", "source_count", "completed_generations", "usage")}), flush=True)
                    last_report = time.monotonic()
        except KeyboardInterrupt:
            self.stop("Keyboard interrupt; resume to continue")
        except Exception as exc:
            self.stop(str(exc).replace(self.key, "[REDACTED]"))
        finally:
            # Allow in-flight responses to save before exiting.
            executor.shutdown(wait=True, cancel_futures=True)
        complete = len(self.results) == len(self.selection["samples"]) and all(
            r["status"] in TERMINAL for r in self.results.values())
        if self.usage()["missing_cost_count"]:
            self.stop("Missing API cost; saved output retained, reconcile before resuming")
        return self.summary("paused" if self.reason else "completed" if complete else "partial")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, default=DATASET)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--variants-per-scene", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--max-sources", type=int, help="Process at most N unfinished sources in this invocation")
    args = parser.parse_args()
    if (not 1 <= args.variants_per_scene <= 100 or not 1 <= args.concurrency <= 32
            or (args.max_sources is not None and args.max_sources < 1)):
        parser.error("Use 1-100 variants, 1-32 concurrent requests and positive max-sources")
    if args.execute and (args.max_cost_usd is None or not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0):
        parser.error("--execute requires a positive --max-cost-usd reported-cost threshold")
    if args.resume and args.run_dir:
        parser.error("Choose --resume or --run-dir")
    run = resolve_path(args.resume or args.run_dir or ROOT / "runs/generation" / dt.datetime.now(
        dt.timezone.utc).strftime("bulk-openrouter-%Y%m%dT%H%M%SZ")).resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process is using this run") from exc
        if args.resume and not (run / "selection.json").exists():
            raise ValueError("No frozen selection to resume")
        manifest_path = resolve_path(args.dataset_manifest).resolve()
        if args.resume:
            old = read_json(run / "selection.json")
            if old.get("implementation_sha256") != implementation_hashes():
                raise ValueError("Generation implementation changed; refusing mixed resume")
            manifest_path = resolve_path(old["dataset_manifest"])
            args.variants_per_scene = old["variants_per_scene"]
        selection = build_selection(manifest_path, args.variants_per_scene)
        selection_path = run / "selection.json"
        if selection_path.exists():
            if read_json(selection_path) != selection:
                raise ValueError("Frozen source, split, settings, exclusions or code changed; refusing mixed resume")
        else:
            atomic_json(selection_path, selection)
        plan = frozen_plan(selection, args.concurrency)
        if not (run / "plan.json").exists():
            atomic_json(run / "plan.json", plan)
        if not args.execute:
            # Intentionally do not read a key or construct a network client in preparation mode.
            if not (run / "summary.json").exists():
                atomic_json(run / "summary.json", {"model": selection["model"], "prompt_version": VERSION,
                            "status": "prepared_awaiting_user_confirmation", "completed_count": 0, "results": []})
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            print(f"Prepared only. No model API calls. Run: {run}")
            return
        atomic_json(run / "execution.json", {"pid": os.getpid(), "concurrency": args.concurrency,
                    "max_cost_usd": args.max_cost_usd, "max_sources": args.max_sources,
                    "started_at": dt.datetime.now(dt.timezone.utc).isoformat()})
        result = Batch(run, selection, args.concurrency, args.max_cost_usd, load_key()).execute(args.max_sources)
        print(f"Status: {result['status']}. Details: {run / 'summary.json'}")
        if result["status"] == "paused":
            raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
