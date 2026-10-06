"""Run a bounded Task 2 extraction → generation → review pilot."""

import argparse
import datetime as dt
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT, resolve_path
from customer_policy_verifier.data.prepare_data import RAW, SPLIT, load_rows


MODEL = "qwen/qwen3.7-flash"
PROMPT_VERSION = "pilot-v12"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
TURN_PREFIX = re.compile(r"^turn \d+: (?:agent|customer):")
PLACEHOLDER = re.compile(r"<[^<>\n]+>")


def json_schema(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object", "properties": properties,
                "required": required, "additionalProperties": False,
            },
        },
    }


def array_of_strings() -> dict:
    return {"type": "array", "items": {"type": "string"}}


RULE = {"type": "object", "properties": {
    "quote": {"type": "string"}, "meaning": {"type": "string"}},
    "required": ["quote", "meaning"], "additionalProperties": False}
FACT = {"type": "object", "properties": {
    "turn_index": {"type": "integer"}, "quote": {"type": "string"},
    "meaning": {"type": "string"}},
    "required": ["turn_index", "quote", "meaning"], "additionalProperties": False}
EXTRACT_SCHEMA = json_schema("scene_extraction", {
    "rules": {"type": "array", "items": RULE},
    "customer_request": {"type": "string"},
    "known_facts": {"type": "array", "items": FACT},
    "unknowns": array_of_strings(),
    "dialogue_progress": {"type": "string"},
    "response_claims": array_of_strings(),
    "label_rationale": {"type": "string"},
    "can_expand": {"type": "boolean"},
    "issues": array_of_strings(),
}, ["rules", "customer_request", "known_facts", "unknowns", "dialogue_progress",
    "response_claims", "label_rationale", "can_expand", "issues"])
GENERATE_SCHEMA = json_schema("scenario_variant", {
    "conversation": array_of_strings(),
    "candidate_response": {"type": "string"},
    "generation_note": {"type": "string"},
}, ["conversation", "candidate_response", "generation_note"])
PRESERVE_SCHEMA = json_schema("preservation_review", {
    "verdict": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
    "reason": {"type": "string"},
}, ["verdict", "reason"])
JUDGE_SCHEMA = json_schema("independent_policy_judgment", {
    "verdict": {"type": "string", "enum": ["compliant", "violating", "uncertain"]},
    "reason": {"type": "string"},
}, ["verdict", "reason"])


def messages(system: str, payload: dict) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def extraction_request(sample: dict) -> tuple[list[dict], dict]:
    return messages(
        "Extract a compact, traceable scene from the supplied customer-service sample. "
        "For each rule, quote MUST be copied verbatim from ONE entry in policy_quote_options. "
        "Use separate rule entries for condition and action lines; never combine entries, "
        "paraphrase, change spacing, or insert ellipses in quote. Describe the "
        "full condition, AND/OR relationship, exception and action order in meaning, using "
        "only policy. Rule meanings must not assert that an event has happened or add "
        "standard business practices absent from policy. "
        "Only conversation contains past events. Each known_fact must copy a short exact "
        "substring of conversation[turn_index], with a zero-based ARRAY index below "
        "history_length. Distinguish a customer's report from a verified system result. "
        "Receiving identity details alone is not evidence that identity verification completed. "
        "Preserve ambiguous source wording in EVERY summary field, including customer_request, "
        "fact meanings and dialogue_progress. An unstated object, cause or physical form is "
        "unknown; do not fill it using policy or common business assumptions. Describe ONLY "
        "this sample, never unrelated examples from these instructions. "
        "dialogue_progress must contain ONLY the state at the END OF HISTORY, before the "
        "candidate response. The candidate response is pending, even with a 'turn N' prefix: "
        "put its claims only in response_claims and label_rationale, never in known_facts "
        "or dialogue_progress; do not even summarize the pending response in dialogue_progress. "
        "Write dialogue_progress as a brief historical summary with a supporting [history index] "
        "for each claim. If an action occurs only in candidate_response, it has NOT occurred "
        "within the history. For example, when history ends with a status request and the "
        "candidate asks for a name, progress is 'Customer requested status [1]', NOT 'Agent "
        "asked for a name'. Before returning, remove pending-response actions from this field. "
        "Describe erroneous claims without correcting them. In response_claims preserve "
        "the original modality: 'might be wrong' is a tentative possibility, not an assertion "
        "that an error exists; 'let me check' is an intention, not a completed operation. "
        "List decision-relevant unknowns explicitly. A vague 'let me check' expresses an "
        "intention, not a completed lookup or a known lookup target. Do not activate a "
        "membership or order-status branch while its condition is unknown. "
        "Treat expected_label as an annotation to CHECK, not an answer to justify at all costs. "
        "First decide what the current response actually does under policy and history, "
        "then compare with expected_label in label_rationale. Evaluate only "
        "the current candidate response: a correct intermediate step need not complete "
        "the workflow or already include later steps. Asking for identifiers is not a "
        "violation merely because later authentication steps have not happened yet. "
        "Lack of a historical completion statement alone is not proof of a skipped step. "
        "Unknown branch conditions do not establish that a conditional action was mandatory "
        "at this turn. Do not invent the target of a vague check in label_rationale. "
        "Set can_expand=true when the given "
        "label is supported and key facts can be preserved, including clearly violating "
        "responses. Set can_expand=false for an unsupported label or material ambiguity. "
        "Keep label_rationale to 2-3 concise sentences using decisive evidence. "
        "If can_expand=true, issues must be empty: violations supporting a violating label "
        "belong only in label_rationale. Otherwise issues lists concrete source ambiguities "
        "or label conflicts. Keep explanations concise; "
        "write values in English and return JSON only.",
        {**{key: sample[key] for key in ("policy", "conversation", "candidate_response", "expected_label")},
         "policy_quote_options": [line.strip() for line in sample["policy"].splitlines()
                                  if line.strip() and not line.strip().startswith("<")],
         "history_length": len(sample["conversation"])},
    ), EXTRACT_SCHEMA


def generation_request(sample: dict, scene: dict) -> tuple[list[dict], dict]:
    return messages(
        "Rewrite BOTH the conversation history and candidate_response as one fresh English "
        "wording variant. Rephrase customer requests and agent sentences using different wording "
        "or sentence structure. An unchanged history is INVALID even if the reply changes. "
        "Change at least one substantive history turn beyond greetings, casing, punctuation "
        "or whitespace, and rephrase the reply. You may copy short name/ID answers and ambiguous "
        "phrases when needed, but must rewrite other substantive sentences. "
        "Change expression only. The original utterances establish facts and behavior. "
        "extracted_scene is an UNVERIFIED model summary and may contain invented details "
        "or mistakenly include the pending response as history. It has no authority to "
        "change the original conversation. Use it only to understand the intended label; "
        "neither it nor policy can supply new events or resolve ambiguity. Preserve all facts, "
        "requests, claims, omissions, uncertainty, action order and information disclosure. "
        "Preserve AND/OR/IF, negation, exceptions, quantities and timing in each ORIGINAL utterance, "
        "even if policy permits a different choice. Name, account ID AND order ID must remain "
        "ALL THREE, never name OR account ID plus order ID. Unknowns must remain unknown. "
        "For example, keep 'lost my 2fa' unspecified rather than inventing a lost phone/device/token "
        "or locked account. 'Has it shipped?' must remain that open question, not a choice between "
        "shipping states. Preserve intentions versus completed actions and tentative versus "
        "confirmed claims. Preserve action objects: a link remains a link, a credit card remains "
        "a credit card. Do not add verification, lookups, statuses, promises or actions. "
        "Keep the expected label AND the original reason for it. For violating examples retain "
        "the SAME skipped step, erroneous claim or refusal: never correct it, insert an omitted "
        "question or replace the error with another violation. For compliant examples retain "
        "the same current step without adding later steps. Do not put labels or policy judgments "
        "in dialogue. Keep exact turn count, order, 'turn N: agent/customer:' prefixes, placeholders, "
        "names, identifiers and numeric values. If an original turn has an ambiguous phrase "
        "such as 'lost my 2fa', copy that phrase exactly and rewrite the surrounding sentence "
        "or other substantive turns. Never adopt a more specific interpretation from "
        "extracted_scene. Check fidelity against the ORIGINAL turns before returning. "
        "Output ONLY a filled JSON object with exactly conversation (array of rewritten turns), "
        "candidate_response (rewritten turn string), generation_note (brief wording changes). "
        "Do not copy input fields or output a schema or audit report.",
        {"policy": sample["policy"], "conversation": sample["conversation"],
         "candidate_response": sample["candidate_response"], "expected_label": sample["expected_label"],
         "extracted_scene": scene},
    ), GENERATE_SCHEMA


def preservation_request(sample: dict, scene: dict, variant: dict) -> tuple[list[dict], dict]:
    return messages(
        "Compare the original and variant. Pass only if decision-relevant facts, unknowns, "
        "policy applicability, action order, dialogue progress, and the response's core claims "
        "are preserved. This verdict is about fidelity, not policy compliance: if BOTH "
        "responses violate the same rule in the same way, that is compatible with pass. "
        "Fail only for a material change between original and variant. If uncertain, "
        "say uncertain. Keep the verdict consistent with the reason. Return JSON only.",
        {"policy": sample["policy"], "original_conversation": sample["conversation"],
         "original_response": sample["candidate_response"], "scene": scene,
         "variant_conversation": variant["conversation"],
         "variant_response": variant["candidate_response"]},
    ), PRESERVE_SCHEMA


def judgment_request(sample: dict, variant: dict) -> tuple[list[dict], dict]:
    # No original label, scene, generation note or preservation result enters this call.
    return messages(
        "Judge whether this candidate customer-service response follows the supplied policy "
        "given only the conversation so far. Do not assume unmentioned facts or future actions. "
        "Choose compliant, violating, or uncertain and explain using the supplied material. "
        "Return JSON only.",
        {"policy": sample["policy"], "conversation": variant["conversation"],
         "candidate_response": variant["candidate_response"]},
    ), JUDGE_SCHEMA


def check_extraction(sample: dict, scene: dict) -> list[str]:
    issues = []
    if not isinstance(scene, dict) or not scene.get("rules") or not scene.get("known_facts"):
        return ["missing rule or fact evidence"]
    for rule in scene["rules"]:
        if not isinstance(rule, dict) or not rule.get("quote") or rule["quote"] not in sample["policy"]:
            issues.append("rule quote missing from policy")
    for fact in scene["known_facts"]:
        index = fact.get("turn_index") if isinstance(fact, dict) else None
        quote = fact.get("quote") if isinstance(fact, dict) else None
        if not isinstance(index, int) or not 0 <= index < len(sample["conversation"]):
            issues.append("invalid fact turn index")
        elif not isinstance(quote, str) or not quote or quote not in sample["conversation"][index]:
            issues.append("fact quote missing from cited turn")
    return issues


def check_variant(sample: dict, variant: dict) -> list[str]:
    issues = []
    if not isinstance(variant, dict) or not isinstance(variant.get("conversation"), list):
        return ["missing variant conversation"]
    turns = variant["conversation"]
    if len(turns) != len(sample["conversation"]):
        issues.append("turn count changed")
    for original, new in zip(sample["conversation"], turns):
        if not isinstance(new, str) or not TURN_PREFIX.match(new):
            issues.append("invalid turn prefix")
        elif TURN_PREFIX.match(original).group() != TURN_PREFIX.match(new).group():
            issues.append("turn index or speaker changed")
    response = variant.get("candidate_response")
    if not isinstance(response, str) or not TURN_PREFIX.match(response):
        issues.append("invalid candidate response prefix")
    elif TURN_PREFIX.match(sample["candidate_response"]).group() != TURN_PREFIX.match(response).group():
        issues.append("candidate response turn changed")
    if sorted(PLACEHOLDER.findall("\n".join(sample["conversation"]))) != sorted(
        PLACEHOLDER.findall("\n".join(s for s in turns if isinstance(s, str)))
    ):
        issues.append("conversation placeholders changed")
    if isinstance(response, str) and sorted(PLACEHOLDER.findall(sample["candidate_response"])) != sorted(
        PLACEHOLDER.findall(response)
    ):
        issues.append("candidate response placeholders changed")
    if turns == sample["conversation"]:
        issues.append("conversation unchanged")
    elif len(turns) == len(sample["conversation"]) and all(
        isinstance(new, str) and " ".join(original.split()) == " ".join(new.split())
        for original, new in zip(sample["conversation"], turns)
    ):
        issues.append("conversation only whitespace changed")
    if turns == sample["conversation"] and response == sample["candidate_response"]:
        issues.append("exact duplicate")
    return issues


def select_samples(rows: list[dict], split: dict) -> list[dict]:
    selected = []
    used_groups = set()
    for label, response_key in (("compliant", "orig_gt_response"), ("violating", "manipulated_gt_response")):
        choices = []
        for index, row in enumerate(rows):
            if split["group_assignment"][str(row["orig_conv_id"])] != "train":
                continue
            if not TURN_PREFIX.match(row[response_key]):
                continue
            digest = hashlib.sha256(f"{split['seed']}:{index}:{label}".encode()).hexdigest()
            choices.append((digest, index, row))
        for _, index, row in sorted(choices):
            if row["orig_conv_id"] in used_groups:
                continue
            used_groups.add(row["orig_conv_id"])
            selected.append({
                "source_row_index": index, "orig_conv_id": row["orig_conv_id"],
                "intent": row["intent"], "policy": row["instruction"],
                "conversation": row["conversation"], "candidate_response": row[response_key],
                "expected_label": label,
            })
            break
    if len(selected) != 2:
        raise ValueError("Could not select two valid, distinct training groups")
    return selected


def load_key() -> str:
    import os
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.partition("=")[2].strip().strip('"\'')
                    break
    if not key:
        raise ValueError("OPENROUTER_API_KEY is empty. Add it to .env or the environment.")
    return key


def wire_request(model: str, request_messages: list[dict], schema: dict,
                 json_mode: bool = False) -> tuple[list[dict], dict]:
    if model == "qwen/qwen3.7-flash" or json_mode:
        # OpenRouter advertises JSON mode for this model, but not schema enforcement.
        hints = {
            "scene_extraction": "Required keys: rules (array of objects with quote and meaning), "
            "customer_request (string), known_facts (array of objects with turn_index integer, "
            "quote and meaning), unknowns (array of strings), dialogue_progress (string), "
            "response_claims (array of strings), label_rationale (string), can_expand (boolean), "
            "issues (array of strings).",
            "scenario_variant": "Required keys: conversation (array of complete turn strings), "
            "candidate_response (complete turn string), generation_note (string).",
            "preservation_review": "Required keys: verdict (pass, fail, or uncertain), reason (string).",
            "independent_policy_judgment": "Required keys: verdict (compliant, violating, or uncertain), reason (string).",
        }
        hint = hints[schema["json_schema"]["name"]]
        request_messages = [
            {**request_messages[0], "content": request_messages[0]["content"]
             + " Return a FILLED JSON object, never a schema definition. "
             + "Do not use keys named type, properties, required or additionalProperties. " + hint},
            *request_messages[1:],
        ]
        return request_messages, {"type": "json_object"}
    return request_messages, schema


def schema_issues(value: object, schema: dict, path: str = "$") -> list[str]:
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int}
    if kind in types and (not isinstance(value, types[kind]) or
                          (kind == "integer" and isinstance(value, bool))):
        return [f"{path} should be {kind}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path} has an invalid value"]
    if kind == "object":
        problems = [f"{path}.{key} is missing" for key in schema.get("required", []) if key not in value]
        if schema.get("additionalProperties") is False:
            problems += [f"{path}.{key} is unexpected" for key in value if key not in schema["properties"]]
        for key, spec in schema.get("properties", {}).items():
            if key in value:
                problems += schema_issues(value[key], spec, f"{path}.{key}")
        return problems
    if kind == "array":
        return [problem for index, item in enumerate(value)
                for problem in schema_issues(item, schema["items"], f"{path}[{index}]")]
    return []


def call_model(key: str, model: str, request_messages: list[dict], response_format: dict,
               schema: dict) -> tuple[dict, dict]:
    body = {"model": model, "messages": request_messages, "response_format": response_format,
            "provider": {"require_parameters": True}, "temperature": 0.2, "max_tokens": 5000}
    if model == "qwen/qwen3.7-flash":
        body["reasoning"] = {"enabled": False}
        body["max_tokens"] = 1800
    request = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    choice = payload["choices"][0]
    content = choice["message"]["content"]
    problems = []
    if choice.get("finish_reason") != "stop":
        problems.append(f"finish_reason={choice.get('finish_reason')}")
    if not isinstance(content, str):
        content = ""
    try:
        result = json.loads(content)
        problems += schema_issues(result, schema["json_schema"]["schema"])
    except json.JSONDecodeError:
        result = {"unparsed_content": content}
        problems.append("model output is not valid JSON")
    return result, {"id": payload.get("id"), "model": payload.get("model"),
                    "usage": payload.get("usage", {}), "finish_reason": choice.get("finish_reason"),
                    "schema_issues": problems}


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Make up to 8 paid API calls for two samples")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if not SPLIT.exists():
        raise SystemExit("Run python scripts/cpv.py data prepare first to freeze the split")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    if hashlib.sha256(RAW.read_bytes()).hexdigest() != split["source_sha256"]:
        raise SystemExit("Raw file differs from the frozen split")
    selected = select_samples(load_rows(RAW), split)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = resolve_path(args.run_dir) if args.run_dir else ROOT / "runs/generation" / f"pilot-{stamp}"
    key = load_key() if args.execute else None
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "selection.json", {"prompt_version": PROMPT_VERSION, "model": args.model,
                                           "source_sha256": split["source_sha256"], "samples": selected})
    for number, sample in enumerate(selected):
        request_messages, schema = extraction_request(sample)
        request_messages, response_format = wire_request(args.model, request_messages, schema)
        save_json(run_dir / f"sample-{number}-extract-request.json", {"messages": request_messages,
                                                                      "response_format": response_format})
    if not args.execute:
        print(f"Dry run: selected 2 training samples and wrote extraction requests to {run_dir}")
        return
    summary = []
    try:
        for number, sample in enumerate(selected):
            record = {"source_row_index": sample["source_row_index"],
                      "orig_conv_id": sample["orig_conv_id"], "expected_label": sample["expected_label"]}
            request_messages, schema = extraction_request(sample)
            request_messages, response_format = wire_request(args.model, request_messages, schema)
            scene, meta = call_model(key, args.model, request_messages, response_format, schema)
            save_json(run_dir / f"sample-{number}-extract.json", {"output": scene, "api": meta})
            problems = meta["schema_issues"] or check_extraction(sample, scene)
            if problems or not scene.get("can_expand"):
                reported_issues = scene.get("issues", [])
                if not isinstance(reported_issues, list):
                    reported_issues = []
                record.update(status="rejected_extraction", reasons=problems + reported_issues)
                summary.append(record)
                continue
            request_messages, schema = generation_request(sample, scene)
            request_messages, response_format = wire_request(args.model, request_messages, schema)
            save_json(run_dir / f"sample-{number}-generate-request.json", {"messages": request_messages,
                                                                           "response_format": response_format})
            variant, meta = call_model(key, args.model, request_messages, response_format, schema)
            save_json(run_dir / f"sample-{number}-generate.json", {"output": variant, "api": meta})
            problems = meta["schema_issues"] or check_variant(sample, variant)
            if problems:
                record.update(status="rejected_format", reasons=problems)
                summary.append(record)
                continue
            request_messages, schema = preservation_request(sample, scene, variant)
            request_messages, response_format = wire_request(args.model, request_messages, schema)
            save_json(run_dir / f"sample-{number}-preserve-request.json", {"messages": request_messages,
                                                                           "response_format": response_format})
            preservation, meta = call_model(key, args.model, request_messages, response_format, schema)
            save_json(run_dir / f"sample-{number}-preserve.json", {"output": preservation, "api": meta})
            if meta["schema_issues"]:
                record.update(status="rejected_review_format", reasons=meta["schema_issues"])
                summary.append(record)
                continue
            request_messages, schema = judgment_request(sample, variant)
            request_messages, response_format = wire_request(args.model, request_messages, schema)
            save_json(run_dir / f"sample-{number}-judge-request.json", {"messages": request_messages,
                                                                        "response_format": response_format})
            judgment, meta = call_model(key, args.model, request_messages, response_format, schema)
            save_json(run_dir / f"sample-{number}-judge.json", {"output": judgment, "api": meta})
            if meta["schema_issues"]:
                record.update(status="rejected_review_format", reasons=meta["schema_issues"])
                summary.append(record)
                continue
            if preservation.get("verdict") == "pass" and judgment.get("verdict") == sample["expected_label"]:
                record.update(status="candidate_for_human_review", variant={
                    "policy": sample["policy"], "conversation": variant["conversation"],
                    "candidate_response": variant["candidate_response"], "label": sample["expected_label"]})
            else:
                record.update(status="rejected_review", preservation=preservation, judgment=judgment)
            summary.append(record)
    finally:
        save_json(run_dir / "summary.json", {"model": args.model, "prompt_version": PROMPT_VERSION,
                                            "results": summary})
    print(f"Pilot complete; review {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
