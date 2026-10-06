"""Offline data validation and decision-format adapters, without torch dependencies."""

import collections
import hashlib
import json
import math
from pathlib import Path


LABELS = ("compliant", "violating")
# Adapted from SemIf-OpenJev core.py/direct.py, commit
# 23cf1f39fc9534fe81437200959b6dfc7106e45a (MIT; see LICENSES).
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
OPTION_LETTERS = ("A", "B")
PROMPT_VERSION = "semif-direct-v1"
QUESTION = (
    "Does the candidate customer-service response comply with the policy at the "
    "current conversation step? Judge the response using the policy and the "
    "conversation history; a valid intermediate step need not complete the entire workflow."
)
CANDIDATES = (
    "Compliant: the candidate response follows the policy requirements that apply "
    "at the current conversation step.",
    "Violating: the candidate response violates an applicable policy requirement, "
    "including a required condition, action, or order of steps.",
)
MODEL_FIELDS = ("policy", "conversation", "candidate_response")
INPUT_NORMALIZATIONS = ("none", "lowercase")
FILE_NAMES = {
    "train": "train.jsonl",
    "train_original": "train_original.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _input_values(record):
    if not isinstance(record, dict):
        raise ValueError("A dataset record must be an object")
    for key in ("policy", "candidate_response"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"Missing or invalid {key}")
    turns = record.get("conversation")
    if not isinstance(turns, list) or not turns or any(not isinstance(t, str) or not t.strip() for t in turns):
        raise ValueError("Missing or invalid conversation turns")
    return {key: record[key] for key in MODEL_FIELDS}


def model_input_values(record, input_normalization="none"):
    """Apply the same label-independent transform to training and inference evidence."""
    if input_normalization not in INPUT_NORMALIZATIONS:
        raise ValueError(f"Unknown input_normalization: {input_normalization}")
    values = _input_values(record)
    if input_normalization == "lowercase":
        return {key: ([turn.lower() for turn in value] if isinstance(value, list)
                      else value.lower()) for key, value in values.items()}
    return values


def state_text(record, input_normalization="none"):
    """Serialize only the three model-input fields, with stable JSON formatting."""
    return json.dumps(model_input_values(record, input_normalization), ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"))


def _fingerprint(record, input_normalization="none"):
    values = model_input_values(record, input_normalization)
    normalized = {key: ([" ".join(t.split()) for t in value] if isinstance(value, list)
                        else " ".join(value.split())) for key, value in values.items()}
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_metadata(record):
    if not isinstance(record.get("id"), str) or not record["id"].strip():
        raise ValueError("Missing or invalid record id")
    if record.get("label") not in LABELS:
        raise ValueError(f"Unknown label for {record['id']}")
    meta = record.get("metadata")
    if not isinstance(meta, dict):
        raise ValueError(f"Missing metadata for {record['id']}")
    group = meta.get("orig_conv_id")
    if type(group) not in (str, int) or not str(group).strip():
        raise ValueError(f"Missing or invalid conversation group for {record['id']}")
    if not isinstance(meta.get("intent"), str) or not meta["intent"].strip():
        raise ValueError(f"Missing or invalid intent for {record['id']}")
    return meta


def _read_records(payload, name):
    try:
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSONL in {name}: {exc}") from exc
    if not rows:
        raise ValueError(f"Empty dataset file: {name}")
    return rows


def _validate_rows(rows, name, source_digest):
    partition = "train" if name == "train_original" else name
    ids, input_labels = set(), {}
    for record in rows:
        _input_values(record)
        meta = _record_metadata(record)
        if record["id"] in ids:
            raise ValueError(f"Duplicate record id in {name}: {record['id']}")
        ids.add(record["id"])
        if meta.get("split") != partition:
            raise ValueError(f"Record partition mismatch in {name}: {record['id']}")
        if meta.get("source_type") not in ("original", "synthetic"):
            raise ValueError(f"Invalid source type in {name}: {record['id']}")
        if name != "train" and meta["source_type"] != "original":
            raise ValueError(f"Only original records are allowed in {name}")
        if meta.get("source_sha256") != source_digest:
            raise ValueError(f"Source hash mismatch in {name}: {record['id']}")
        key = _fingerprint(record)
        if key in input_labels and input_labels[key] != record["label"]:
            raise ValueError(f"Conflicting labels for the same complete input in {name}")
        input_labels[key] = record["label"]


def _check_file_stats(rows, payload, spec, name):
    if not isinstance(spec, dict) or not _valid_digest(spec.get("sha256")):
        raise ValueError(f"Missing or invalid manifest file digest: {name}")
    if _sha256(payload) != spec["sha256"]:
        raise ValueError(f"File hash mismatch: {name}")
    if type(spec.get("count")) is not int or spec["count"] != len(rows):
        raise ValueError(f"File count mismatch: {name}")
    labels = dict(collections.Counter(r["label"] for r in rows))
    if spec.get("label_counts") != labels:
        raise ValueError(f"Label counts mismatch: {name}")
    optional_stats = {
        "bytes": len(payload),
        "source_type_counts": dict(collections.Counter(r["metadata"]["source_type"] for r in rows)),
        "review_status_counts": dict(collections.Counter(r["metadata"].get("review_status") for r in rows)),
        "conversation_groups": len({str(r["metadata"]["orig_conv_id"]) for r in rows}),
        "intents": len({r["metadata"]["intent"] for r in rows}),
    }
    for key, actual in optional_stats.items():
        if key in spec and spec[key] != actual:
            raise ValueError(f"Manifest {key} mismatch: {name}")


def load_dataset_bundle(data_dir):
    """Read the existing task2-v1 bundle and fail on drift; never split or rewrite it.

    Original validation/test digests come from the build manifest's input records.
    Those records may contain another machine's absolute paths; no source archive
    needs to be present when this portable bundle is copied to a GPU machine.
    """
    directory = Path(data_dir)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dataset manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != "training-data-v1":
        raise ValueError("Expected the training-data-v1 manifest")
    mapping = manifest.get("label_to_id")
    if (mapping != {label: i for i, label in enumerate(LABELS)}
            or any(type(v) is not int for v in mapping.values())):
        raise ValueError("Label mapping must be compliant=0, violating=1")
    if manifest.get("model_input_fields") != list(MODEL_FIELDS) or manifest.get("target_field") != "label":
        raise ValueError("Manifest model input fields or target field changed")
    if type(manifest.get("seed")) is not int or manifest["seed"] != 20261002:
        raise ValueError("Frozen dataset seed changed")
    source_digest = manifest.get("source_sha256")
    if not _valid_digest(source_digest) or not _valid_digest(manifest.get("split_sha256")):
        raise ValueError("Missing or invalid frozen source/split hashes")
    specs = manifest.get("files")
    if not isinstance(specs, dict) or set(specs) != set(FILE_NAMES.values()):
        raise ValueError("Manifest must describe the four fixed dataset files")
    bundle = {"manifest": manifest}
    payloads = {}
    for key, filename in FILE_NAMES.items():
        payload = (directory / filename).read_bytes()
        if not isinstance(specs[filename], dict) or _sha256(payload) != specs[filename].get("sha256"):
            raise ValueError(f"File hash mismatch: {filename}")
        rows = _read_records(payload, filename)
        _validate_rows(rows, key, source_digest)
        _check_file_stats(rows, payload, specs[filename], filename)
        bundle[key], payloads[key] = rows, payload

    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("Manifest is missing frozen original input digests")
    for partition in ("validation", "test"):
        suffix = f"/originals/{partition}.jsonl"
        matching = [entry for entry in inputs if isinstance(entry, dict)
                    and isinstance(entry.get("path"), str)
                    and ("/" + entry["path"].replace("\\", "/")).endswith(suffix)]
        if len(matching) != 1 or matching[0].get("sha256") != _sha256(payloads[partition]):
            raise ValueError(f"Frozen original {partition} file changed")

    baseline = {r["id"]: r for r in bundle["train_original"]}
    mixed_originals = {r["id"]: r for r in bundle["train"] if r["metadata"]["source_type"] == "original"}
    if baseline != mixed_originals:
        raise ValueError("train_original is not exactly the original subset of train")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_rows, right_rows = bundle[left], bundle[right]
        left_groups = {str(r["metadata"]["orig_conv_id"]) for r in left_rows}
        right_groups = {str(r["metadata"]["orig_conv_id"]) for r in right_rows}
        if left_groups & right_groups:
            raise ValueError(f"Conversation groups cross partitions: {left} / {right}")
        if {r["id"] for r in left_rows} & {r["id"] for r in right_rows}:
            raise ValueError(f"Record ids cross partitions: {left} / {right}")
        if {_fingerprint(r) for r in left_rows} & {_fingerprint(r) for r in right_rows}:
            raise ValueError(f"Complete inputs cross partitions: {left} / {right}")
    return bundle


def normalization_audit(bundle, input_normalization="none"):
    """Reject conflicts or partition overlap introduced by evidence normalization."""
    partitions, report = {}, {}
    for split in FILE_NAMES:
        labels_by_input = {}
        digest = hashlib.sha256()
        uppercase_responses = collections.Counter({label: 0 for label in LABELS})
        for record in bundle[split]:
            fingerprint = _fingerprint(record, input_normalization)
            if fingerprint in labels_by_input and labels_by_input[fingerprint] != record["label"]:
                raise ValueError(f"Normalized inputs have conflicting labels in {split}")
            labels_by_input[fingerprint] = record["label"]
            evidence = model_input_values(record, input_normalization)
            response = evidence["candidate_response"]
            uppercase_responses[record["label"]] += int(response != response.lower())
            digest.update((state_text(record, input_normalization) + "\n").encode("utf-8"))
        partitions[split] = set(labels_by_input)
        report[split] = {"count": len(bundle[split]), "unique_inputs": len(labels_by_input),
                         "duplicate_inputs": len(bundle[split]) - len(labels_by_input),
                         "evidence_sha256": digest.hexdigest(),
                         "responses_with_uppercase_by_label": dict(uppercase_responses)}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if partitions[left] & partitions[right]:
            raise ValueError(f"Normalized complete inputs cross partitions: {left} / {right}")
    return {"input_normalization": input_normalization, "integrity": "passed", "splits": report}


def to_question(record, input_normalization="none"):
    """Adapt one classified response into a choice decision without leaking its label."""
    state = state_text(record, input_normalization)
    meta = _record_metadata(record)
    label_id = LABELS.index(record["label"])
    return {
        "id": record["id"],
        "case_id": str(meta["orig_conv_id"]),
        "workflow": meta["intent"],
        "state": state,
        "question": {"type": "choice", "text": QUESTION,
                     "keys": list(LABELS), "candidates": list(CANDIDATES)},
        "target": [float(i == label_id) for i in range(len(LABELS))],
    }


def direct_messages(record, input_normalization="none"):
    """Build SemIf Direct messages from a raw record's three input fields only.

    This helper also accepts inference records without labels or management
    metadata. The serialized evidence is identical to ``state_text(record)``.
    A means compliant and B means violating; neither is a gold answer here.
    """
    payload = {
        "evidence": state_text(record, input_normalization),
        "criterion": QUESTION,
        "options": [
            {"letter": letter, "description": candidate}
            for letter, candidate in zip(OPTION_LETTERS, CANDIDATES)
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _token_ids(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Tokenizer must return a nonempty list of integer token ids")
    return ids


def slot_token_ids(tokenizer):
    """Require exact, distinct, single-token A/B slots, as in SemIf Direct."""
    slots = []
    for letter in OPTION_LETTERS:
        ids = _token_ids(tokenizer, letter)
        if len(ids) != 1 or tokenizer.decode(ids) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        slots.append(ids[0])
    if len(slots) != len(set(slots)):
        raise ValueError("Answer-slot tokens must be distinct; token IDs collide")
    return slots


def encode_record(record, tokenizer, max_tokens=2048, input_normalization="none"):
    """Encode one SemIf Direct chat prompt, with no answer token or truncation.

    Return the training decision plus one ``input_ids`` sequence, fixed A/B
    ``option_token_ids``, and the SHA-256 of the exact rendered chat prompt.
    Gold labels and management metadata are returned outside the model input.
    """
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    decision = to_question(record, input_normalization)
    prompt = tokenizer.apply_chat_template(
        direct_messages(record, input_normalization), tokenize=False,
        add_generation_prompt=True, enable_thinking=False
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Chat template must return a nonempty string")
    ids = _token_ids(tokenizer, prompt)
    if len(ids) > max_tokens:
        raise ValueError(f"Record {record['id']} has a {len(ids)}-token prompt exceeding max_tokens={max_tokens}; no truncation allowed")
    slots = slot_token_ids(tokenizer)
    for letter, slot in zip(OPTION_LETTERS, slots):
        if _token_ids(tokenizer, prompt + letter) != ids + [slot]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
    decision["input_ids"] = ids
    decision["option_token_ids"] = slots
    decision["prompt_sha256"] = _sha256(prompt.encode("utf-8"))
    return decision


def binary_metrics(target_ids, probabilities, threshold=0.5):
    """Measure binary decisions from [P(compliant), P(violating)] distributions.

    Violating is the positive class. Rates with zero denominator are reported
    as 0.0; CE clips a zero true-class probability to 1e-12 for finite reports.
    """
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be finite and within [0, 1]")
    targets, distributions = list(target_ids), list(probabilities)
    if not targets or len(targets) != len(distributions):
        raise ValueError("Targets and probabilities must have the same nonzero length")
    matrix = [[0, 0], [0, 0]]
    ce, brier = 0.0, 0.0
    for target, probability in zip(targets, distributions):
        if type(target) is not int or target not in (0, 1):
            raise ValueError("Target ids must be integer 0 or 1")
        if not isinstance(probability, (list, tuple)) or len(probability) != 2:
            raise ValueError("Each probability must contain two class probabilities")
        if any(isinstance(p, bool) or not isinstance(p, (int, float))
               or not math.isfinite(p) or not 0 <= p <= 1 for p in probability):
            raise ValueError("Class probabilities must be finite and within [0, 1]")
        if not math.isclose(sum(probability), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Class probabilities must sum to 1")
        prediction = int(probability[1] >= threshold)
        matrix[target][prediction] += 1
        ce -= math.log(max(probability[target], 1e-12))
        brier += sum((p - float(i == target)) ** 2 for i, p in enumerate(probability))
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    def divide(numerator, denominator):
        return numerator / denominator if denominator else 0.0
    f1_positive = divide(2 * tp, 2 * tp + fp + fn)
    f1_negative = divide(2 * tn, 2 * tn + fp + fn)
    n = len(targets)
    return {"n": n, "accuracy": (tn + tp) / n,
            "macro_f1": (f1_negative + f1_positive) / 2,
            "violating_recall": divide(tp, tp + fn),
            "compliant_false_positive_rate": divide(fp, tn + fp),
            "confusion_matrix": matrix, "cross_entropy": ce / n,
            "brier_sum": brier / n}
