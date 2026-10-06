"""Frozen, zero-shot choice evaluation outside the customer-policy domain.

English evidence, criteria and option descriptions are lowercased for every
model. The existing native SemIf Direct readout is extended to A/B/C/D without
changing the trained checkpoint format or introducing another classifier.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import random
from pathlib import Path

from customer_policy_verifier.training.jev_data import DIRECT_SYSTEM
from customer_policy_verifier.training.backbone_download import sha256
from customer_policy_verifier.training.train_jev import save_json

LETTERS = "ABCD"
VERSION = "ood-choice-v1"
RULE_CRITERION = (
    "Decide whether the query is true under the supplied facts and rules. "
    "Apply rules repeatedly. Use the closed-world convention: an unprovable "
    "positive statement is false, and its negation is true. Explicit negative "
    "facts and negative rule conclusions also apply. Choose true or false."
)
LOGIC_CRITERION = "Choose the single best answer to the question using the supplied passage."


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                 for row in rows), encoding="utf-8")


def rule_subset(rows, count=1000, seed=20261004):
    """Use each Problog row's own English and label; never join original IDs."""
    if count < 2 or count % 2:
        raise ValueError("Rule subset size must be positive and even")
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = set()
    for row in rows:
        identity = row["id"]
        if identity in seen:
            raise ValueError("Duplicate RuleTaker ID")
        seen.add(identity)
        theory = row["theory_assertion_instance"]
        if theory["exception"] is not None or type(theory["label"]) is not bool:
            raise ValueError("RuleTaker requires a verified boolean label without exception")
        english = row["english"]
        statements = english["theory_statements"]
        query = english["assertion_statement"]
        if not statements or not all(isinstance(s, str) and s.strip() for s in statements):
            raise ValueError("Empty RuleTaker English theory")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Empty RuleTaker query")
        group = identity.rsplit("_", 1)[0]
        label = 0 if theory["label"] else 1
        groups[group][label].append({"id": identity, "group_id": group,
            "context": "\n".join(statements), "query": query, "options": ["True", "False"],
            "label_id": label, "task": "ruletaker"})
    # Restrict to theories with both possible labels, so group selection is
    # independent of the balanced target sequence. One query per theory.
    eligible = sorted(group for group, labels in groups.items() if set(labels) == {0, 1})
    if len(eligible) < count:
        raise ValueError("Insufficient distinct theories with both labels")
    rng = random.Random(seed)
    selected = rng.sample(eligible, count)
    targets = [0] * (count // 2) + [1] * (count // 2)
    rng.shuffle(targets)
    return [rng.choice(groups[group][label]) for group, label in zip(selected, targets)]


def logiqa_rows(rows):
    prepared = []
    for line_number, row in enumerate(rows, 1):
        # Official source IDs are not unique. Preserve every official row.
        identity = f"{row['id']}-row-{line_number:04d}"
        if (type(row["answer"]) is not int or row["answer"] not in range(4)
                or len(row["options"]) != 4
                or not all(isinstance(s, str) and s.strip() for s in
                           [row["text"], row["question"], *row["options"]])):
            raise ValueError("Invalid four-choice LogiQA row")
        prepared.append({"id": identity, "source_id": row["id"], "source_row": line_number,
                         "context": row["text"], "query": row["question"],
                         "options": row["options"], "label_id": row["answer"], "task": "logiqa2"})
    return prepared


def input_quality(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        key = json.dumps([row["context"].lower(), row["query"].lower(),
                          *[s.lower() for s in row["options"]]], ensure_ascii=False)
        groups[key].append(row)
    return {"unique_complete_inputs": len(groups),
            "duplicate_extra_rows": sum(len(v) - 1 for v in groups.values()),
            "conflicting_label_groups": sum(len({r["label_id"] for r in v}) > 1 for v in groups.values())}


def permute(row, order):
    n = len(row["options"])
    if sorted(order) != list(range(n)):
        raise ValueError("Order must be a complete option permutation")
    return {**row, "id": row["id"] + "@" + "".join(map(str, order)), "base_id": row["id"],
            "order": list(order), "options": [row["options"][i] for i in order],
            "label_id": order.index(row["label_id"])}


def diagnostics(rows, count=128, seed=20261004):
    n = len(rows[0]["options"])
    if count % n:
        raise ValueError("Diagnostic sample must balance all gold labels")
    rng = random.Random(seed)
    selected = []
    for label in range(n):
        candidates = sorted((r for r in rows if r["label_id"] == label), key=lambda r: r["id"])
        selected.extend(rng.sample(candidates, count // n))
    selected.sort(key=lambda r: r["id"])
    return [permute(row, list(range(n))[rotation:] + list(range(n))[:rotation])
            for row in selected for rotation in range(1, n)]


def messages(row):
    n = len(row["options"])
    if n not in (2, 4) or row["task"] not in ("ruletaker", "logiqa2"):
        raise ValueError("Unsupported choice task")
    evidence = json.dumps({"context": row["context"].lower(), "query": row["query"].lower()},
                          ensure_ascii=False)
    criterion = RULE_CRITERION if row["task"] == "ruletaker" else LOGIC_CRITERION
    payload = {"evidence": evidence, "criterion": criterion.lower(), "options": [
        {"letter": letter, "description": description.lower()}
        for letter, description in zip(LETTERS[:n], row["options"])]}
    return [{"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def encode(row, tokenizer, max_tokens):
    prompt = tokenizer.apply_chat_template(messages(row), tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"{row['task']}/{row['id']} has {len(ids)} tokens; no truncation allowed")
    slots = []
    for letter in LETTERS[:len(row["options"])]:
        token = tokenizer.encode(letter, add_special_tokens=False)
        if len(token) != 1 or tokenizer.decode(token) != letter:
            raise ValueError("Choice letter must be an exact single token")
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + token:
            raise ValueError("Answer boundary changes tokenization")
        slots.extend(token)
    if len(set(slots)) != len(slots):
        raise ValueError("Answer token collision")
    return {"id": row["id"], "input_ids": ids, "option_token_ids": slots,
            "target": [float(i == row["label_id"]) for i in range(len(slots))],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def metrics(predictions, n):
    if not predictions:
        raise ValueError("Empty evaluation")
    matrix = [[0] * n for _ in range(n)]
    cross_entropy, brier = 0.0, 0.0
    for row in predictions:
        label, probs = row["label_id"], row["probabilities"]
        if (label not in range(n) or len(probs) != n or any(not math.isfinite(p) or p < 0 for p in probs)
                or not math.isclose(sum(probs), 1.0, abs_tol=1e-5)):
            raise ValueError("Invalid prediction distribution")
        predicted = max(range(n), key=probs.__getitem__)
        matrix[label][predicted] += 1
        cross_entropy -= math.log(max(probs[label], 1e-300))
        brier += sum((p - float(i == label)) ** 2 for i, p in enumerate(probs))
    total = len(predictions)
    correct = sum(matrix[i][i] for i in range(n))
    gold_counts = [sum(row) for row in matrix]
    pred_counts = [sum(row[i] for row in matrix) for i in range(n)]
    f1 = [2 * matrix[i][i] / (gold_counts[i] + pred_counts[i])
          if gold_counts[i] + pred_counts[i] else 0.0 for i in range(n)]
    acc, z = correct / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (acc + z * z / (2 * total)) / denominator
    half = z * math.sqrt(acc * (1 - acc) / total + z * z / (4 * total * total)) / denominator
    return {"samples": total, "correct": correct, "accuracy": acc, "macro_f1": sum(f1) / n,
            "accuracy_wilson_95": [center - half, center + half], "confusion_matrix": matrix,
            "gold_letter_counts": dict(zip(LETTERS[:n], gold_counts)),
            "predicted_letter_counts": dict(zip(LETTERS[:n], pred_counts)),
            "uniform_chance_accuracy": 1 / n, "majority_gold_accuracy": max(gold_counts) / total,
            "restricted_cross_entropy": cross_entropy / total, "brier_score": brier / total}


def permutation_summary(primary, variants, n):
    by_id = {p["id"]: p for p in primary}
    grouped = collections.defaultdict(list)
    for row in variants:
        canonical = by_id[row["base_id"]]
        canonical_choice = max(range(n), key=canonical["probabilities"].__getitem__)
        variant_choice = max(range(n), key=row["probabilities"].__getitem__)
        grouped[row["base_id"]].append(row["order"][variant_choice] == canonical_choice)
    return {"base_questions": len(grouped), "variant_predictions": len(variants),
            "semantic_consistency": sum(sum(v) for v in grouped.values()) / len(variants),
            "all_rotations_consistent": sum(all(v) for v in grouped.values()) / len(grouped),
            "rotated_metrics": metrics(variants, n)}


def prepare(root, suite, tokenizer, max_tokens=4096, seed=20261004):
    if suite.exists() and any(suite.iterdir()):
        raise ValueError("Use a fresh suite directory; frozen suites are never overwritten")
    sources = root / "data/evaluation/sources"
    rule_source = sources / "ruletaker/depth-3-problog-test.jsonl"
    logic_source = sources / "logiqa2/test.txt"
    rules = rule_subset(read_jsonl(rule_source), seed=seed)
    logic = logiqa_rows(read_jsonl(logic_source))
    if len(logic) != 1572:
        raise ValueError("Expected all 1,572 official LogiQA 2.0 English test rows")
    suite.mkdir(parents=True, exist_ok=True)
    manifest = {"version": VERSION, "seed": seed, "max_tokens": max_tokens,
        "input_normalization": "lowercase", "readout": "native next-token restricted choice softmax",
        "thinking": False, "system": DIRECT_SYSTEM, "criteria": {"ruletaker": RULE_CRITERION,
        "logiqa2": LOGIC_CRITERION}, "zero_shot": True, "training": "none",
        "pretraining_overlap": "unknown; public benchmarks may occur in Qwen pretraining",
        "quality_audit": {"ruletaker": input_quality(rules), "logiqa2": input_quality(logic)},
        "sources": {"ruletaker": {"url": "https://github.com/allenai/ruletaker",
            "archive_url": "https://aristo-data-public.s3-us-west-2.amazonaws.com/ruletaker/rule-reasoning-dataset-V2020.2.5.zip",
            "archive_sha256": sha256(sources / "ruletaker/rule-reasoning-dataset-V2020.2.5.zip"),
            "member": "rule-reasoning-dataset-V2020.2.5.0/problog/depth-3/test.jsonl",
            "sha256": sha256(rule_source), "sampling": "1,000 distinct theories with both labels; 500 true / 500 false"},
            "logiqa2": {**json.loads((sources / "logiqa2/source.json").read_text()),
            "project_url": "https://github.com/csitfun/LogiQA2.0", "license": "CC BY-NC-SA 4.0",
            "sampling": "complete official English test split"}}, "files": {}}
    for task, primary in (("ruletaker", rules), ("logiqa2", logic)):
        for kind, rows in (("primary", primary), ("permuted", diagnostics(primary, seed=seed))):
            name = f"{task}-{kind}.jsonl"
            write_jsonl(suite / name, rows)
            encoded = [encode(row, tokenizer, max_tokens) for row in rows]
            prompt_digest = hashlib.sha256("".join(r["prompt_sha256"] for r in encoded).encode()).hexdigest()
            manifest["files"][name] = {"sha256": sha256(suite / name), "samples": len(rows),
                "max_prompt_tokens": max(len(r["input_ids"]) for r in encoded),
                "prompt_sequence_sha256": prompt_digest,
                "gold_counts": dict(collections.Counter(r["label_id"] for r in rows))}
    save_json(suite / "manifest.json", manifest)
    return manifest


def load_suite(suite, tokenizer):
    manifest = json.loads((suite / "manifest.json").read_text())
    if (manifest["version"] != VERSION or manifest["system"] != DIRECT_SYSTEM
            or manifest["criteria"] != {"ruletaker": RULE_CRITERION, "logiqa2": LOGIC_CRITERION}):
        raise ValueError("Frozen suite prompt protocol differs from current code")
    result = {}
    for name, expected in manifest["files"].items():
        if sha256(suite / name) != expected["sha256"]:
            raise ValueError("Frozen suite data hash mismatch")
        rows = read_jsonl(suite / name)
        encoded = [encode(row, tokenizer, manifest["max_tokens"]) for row in rows]
        digest = hashlib.sha256("".join(r["prompt_sha256"] for r in encoded).encode()).hexdigest()
        if digest != expected["prompt_sequence_sha256"] or len(rows) != expected["samples"]:
            raise ValueError("Frozen suite tokenizer/prompt/count mismatch")
        result[name] = (rows, encoded)
    return manifest, result
