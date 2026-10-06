"""Full-parameter SemIf Direct training for the frozen customer-policy dataset.

Prompt and native answer-slot readout follow SemIf-OpenJev at
23cf1f39fc9534fe81437200959b6dfc7106e45a. That reference implements inference;
this project adds restricted-choice cross-entropy supervision and training.
Qwen's original pretrained LM head is retained. No text decoding or new head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from contextlib import nullcontext
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT
from customer_policy_verifier.training import jev_data
from customer_policy_verifier.training.backbone_download import (
    download,
    sha256,
    source_digest,
    validate_backbone,
)
from customer_policy_verifier.training.jev_data import (
    CANDIDATES,
    LABELS,
    QUESTION,
    binary_metrics,
    encode_record,
    load_dataset_bundle,
)

DEFAULT_CONFIG = ROOT / "configs" / "jev-training.json"
DEFAULT_DATA = ROOT / "data" / "training" / "task2-v1"
DEFAULT_BACKBONE = ROOT / "models" / "Qwen3-0.6B-modelscope"
DEFAULT_PACKAGE = ROOT / "runs" / "packages" / "autodl" / "customer-policy-verifier-training.tar.gz"
CHECKPOINT_FORMAT = "customer-policy-semif-direct-v1"
CHECKPOINT_SERIALIZATION = "safetensors-json-v1"
CHECKPOINT_NAME = "best.safetensors"


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_execution(cfg):
    if cfg["device"] not in ("cpu", "cuda") or cfg["precision"] not in ("fp32", "bf16"):
        raise ValueError("supported execution: cpu/fp32 or cuda/fp32,bf16")
    if cfg["device"] == "cpu" and cfg["precision"] != "fp32":
        raise ValueError("CPU execution requires --precision fp32")
    if type(cfg["eval_batch_size"]) is not int or cfg["eval_batch_size"] < 1:
        raise ValueError("eval_batch_size must be a positive integer")


def config_for(args):
    cfg = json.loads(Path(args.config).read_text())
    validate_protocol_config(cfg)
    cfg.setdefault("max_steps", -1)
    cfg.setdefault("num_train_epochs", 3)
    for key in ("max_steps", "microbatch_questions", "gradient_accumulation", "eval_batch_size",
                "num_train_epochs", "eval_every", "max_tokens", "device", "precision"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if getattr(args, "num_train_epochs", None) is not None:
        cfg["max_steps"] = -1
    if type(cfg["max_steps"]) is not int or (cfg["max_steps"] != -1 and cfg["max_steps"] < 1):
        raise ValueError("max_steps must be a positive integer or -1 for epoch-based training")
    for key in ("num_train_epochs", "microbatch_questions", "gradient_accumulation", "eval_batch_size",
                "eval_every", "max_tokens"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "grad_clip"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"invalid {key}")
    for key in ("weight_decay",):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"invalid {key}")
    warmup_steps, warmup_ratio = cfg.get("warmup_steps", 0), cfg.get("warmup_ratio", 0.0)
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if (isinstance(warmup_ratio, bool) or not isinstance(warmup_ratio, (int, float))
            or not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio <= 1):
        raise ValueError("warmup_ratio must be finite and within [0, 1]")
    if warmup_steps > 0 and warmup_ratio > 0:
        raise ValueError("use warmup_steps or warmup_ratio, not both; fixed steps override the ratio")
    if cfg.get("eval_strategy", "steps") not in ("steps", "epoch"):
        raise ValueError("eval_strategy must be steps or epoch")
    for key in ("save_total_limit", "logging_every"):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    report_to = cfg.get("report_to", ["tensorboard"])
    if not isinstance(report_to, list) or any(value != "tensorboard" for value in report_to):
        raise ValueError("report_to must be [] or ['tensorboard']")
    validate_execution(cfg)
    if not 0 <= cfg["violating_threshold"] <= 1:
        raise ValueError("invalid violating_threshold")
    return cfg


def validate_protocol_config(cfg):
    if (cfg.get("version") != CHECKPOINT_FORMAT or cfg.get("readout") != "semif-direct"
            or cfg.get("option_letters") != ["A", "B"]
            or cfg.get("objective") != "restricted-choice cross entropy"):
        raise ValueError("Expected SemIf Direct v1 configuration; legacy AgentJev settings are incompatible")
    legacy = {"set_dim", "set_layers", "set_heads", "backbone_lr", "head_lr", "brier_weight"} & cfg.keys()
    if legacy:
        raise ValueError(f"Remove legacy AgentJev configuration fields: {sorted(legacy)}")
    if cfg.get("input_normalization", "none") not in jev_data.INPUT_NORMALIZATIONS:
        raise ValueError("input_normalization must be none or lowercase")


def prompt_protocol(tokenizer, cfg=None):
    """Bind checkpoints to the answer slots and actual tokenizer chat template."""
    protocol = {"version": jev_data.PROMPT_VERSION,
            "system": jev_data.DIRECT_SYSTEM,
            "question": QUESTION, "candidates": list(CANDIDATES),
            "option_letters": list(jev_data.OPTION_LETTERS),
            "option_token_ids": jev_data.slot_token_ids(tokenizer),
            "chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
            "add_generation_prompt": True, "enable_thinking": False,
            "readout": "native last-position vocabulary logits restricted to A/B"}
    normalization = (cfg or {}).get("input_normalization", "none")
    if normalization != "none":
        protocol["input_normalization"] = {"name": normalization, "version": "evidence-lowercase-v1",
                                            "fields": list(jev_data.MODEL_FIELDS), "operation": "str.lower"}
    return protocol


def input_audit(rows):
    lengths = sorted(len(row["input_ids"]) for row in rows)
    return {"questions": len(rows), "prompts": len(lengths), "max_tokens": lengths[-1],
            "median_tokens": lengths[len(lengths) // 2],
            "p95_tokens": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))],
            "truncated": 0}


def package_training(args):
    """Make a portable source/documentation upload with the fixed training data."""
    output = Path(args.output)
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise ValueError("package output already exists; choose a fresh --output")
    project_files = ["README.md", ".gitignore", ".env.example", "pyproject.toml"]
    project_files += [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "scripts").glob("*.py"))]
    project_files += [path.relative_to(ROOT).as_posix() for path in
                      sorted((ROOT / "requirements").glob("*.txt")) if path.is_file()]
    project_files += [path.relative_to(ROOT).as_posix() for path in
                      sorted((ROOT / "docs").rglob("*.md")) if path.is_file()]
    project_files += [path.relative_to(ROOT).as_posix() for path in
                      sorted((ROOT / "configs").rglob("*.json"))
                      if path.is_file() and path != DEFAULT_CONFIG]
    project_files += [path.relative_to(ROOT).as_posix() for path in
                      sorted((ROOT / "tests").glob("*.py")) if path.is_file()]
    project_files += [path.relative_to(ROOT).as_posix() for path in
                      sorted((ROOT / ".github").rglob("*.yml")) if path.is_file()]
    project_files += [name for name in ("data/README.md", "data/catalog.json",
                                      "runs/README.md", "runs/catalog.json")
                      if (ROOT / name).is_file()]
    project_files += [path.relative_to(ROOT).as_posix() for path in
                      sorted((ROOT / "src" / "customer_policy_verifier").rglob("*.py"))
                      if path.is_file()]
    project_files += [path.relative_to(ROOT).as_posix() for path in
                      sorted((ROOT / "LICENSES").rglob("*")) if path.is_file()]
    members = [(ROOT / name, name) for name in project_files]
    members.append((Path(args.config), "configs/jev-training.json"))
    for name in ("manifest.json", "train.jsonl", "train_original.jsonl", "validation.jsonl", "test.jsonl"):
        members.append((Path(args.data_dir) / name, "data/training/task2-v1/" + name))
    file_hashes = {name: sha256(path) for path, name in members}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for path, name in members:
            archive.add(path, arcname="customer-policy-verifier/" + name, recursive=False)
    temporary.replace(output)
    result = {"path": str(output.resolve()), "bytes": output.stat().st_size,
              "sha256": sha256(output), "files": file_hashes}
    save_json(output.with_suffix(output.suffix + ".manifest.json"), result)
    return result


class TinyTokenizer:
    """Offline numerical integration check only; never a production tokenizer."""
    pad_token_id = 0
    eos_token_id = 1
    chat_template = "offline-tiny-chat-v1; system/user/assistant; thinking disabled"

    def encode(self, text, add_special_tokens=False):
        return [({"A": 2, "B": 3}.get(word, 4 + (sum(word.encode()) % 124)))
                for word in text.split()]

    def decode(self, ids):
        return {2: "A", 3: "B"}.get(ids[0], "?") if len(ids) == 1 else "?"

    def get_chat_template(self):
        return self.chat_template

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        if tokenize or not add_generation_prompt or enable_thinking:
            raise ValueError("TinyTokenizer only supports the offline Direct prompt fixture")
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


def tokenizer_for(args, cfg):
    if getattr(args, "tiny_model", False):
        return TinyTokenizer()
    from transformers import AutoTokenizer

    validate_backbone(args.backbone, cfg)
    return AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)


def collate(rows, pad, device):
    import torch

    if not rows or any(not row["input_ids"] for row in rows):
        raise ValueError("Cannot collate empty prompts")
    width = max(len(row["input_ids"]) for row in rows)
    cmax = max(len(row["target"]) for row in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(rows), cmax), dtype=torch.bool, device=device)
    target = torch.zeros((len(rows), cmax), dtype=torch.float32, device=device)
    slots = torch.zeros((len(rows), cmax), dtype=torch.long, device=device)
    for qi, row in enumerate(rows):
        if len(row["option_token_ids"]) != len(row["target"]):
            raise ValueError("Answer-slot count differs from target count")
        mask[qi, :len(row["target"])] = True
        target[qi, :len(row["target"])] = torch.tensor(row["target"], device=device)
        slots[qi, :len(row["target"])] = torch.tensor(row["option_token_ids"], device=device)
        # Every row ends in its final real prompt token, as in upstream Direct.
        ids[qi, -len(row["input_ids"]):] = torch.tensor(row["input_ids"], device=device)
        attention[qi, -len(row["input_ids"]):] = 1
    positions = (attention.cumsum(-1) - 1).clamp_min(0)
    return {"input_ids": ids, "attention_mask": attention, "cand_mask": mask, "target": target,
            "position_ids": positions, "option_token_ids": slots}


def amp_context(cfg):
    if cfg["precision"] == "fp32":
        return nullcontext()
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def evaluate_rows(model, rows, pad, cfg):
    import torch

    model.eval()
    predictions = []
    for start in range(0, len(rows), cfg["eval_batch_size"]):
        chunk = rows[start:start + cfg["eval_batch_size"]]
        batch = collate(chunk, pad, cfg["device"])
        with torch.inference_mode(), amp_context(cfg):
            probabilities = model(batch)["logits"].float().softmax(-1).cpu().tolist()
        for row, prob in zip(chunk, probabilities):
            predictions.append({"id": row["id"], "case_id": row["case_id"], "workflow": row["workflow"],
                                "label_id": row["target"].index(1.0), "probabilities": prob})
    return binary_metrics([p["label_id"] for p in predictions],
                          [p["probabilities"] for p in predictions], cfg["violating_threshold"]), predictions


def build_model(args, cfg):
    import torch

    from customer_policy_verifier.training.jev_model import SemIfDirectModel

    if getattr(args, "tiny_model", False):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        base = Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                    head_dim=8, max_position_embeddings=cfg["max_tokens"],
                                    tie_word_embeddings=True))
        model = SemIfDirectModel(base)
    else:
        validate_backbone(args.backbone, cfg, require_weights=True)
        model = SemIfDirectModel.from_pretrained(args.backbone, local_files_only=True,
                                                attn_implementation=cfg["attn_implementation"])
    if cfg["device"] == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA GPU unavailable; use an AutoDL CUDA instance")
        if cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("this GPU cannot run bf16; select --precision fp32")
    if cfg["gradient_checkpointing"]:
        model.causal_lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.causal_lm.config.use_cache = False
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("all original Qwen CausalLM parameters must be trainable")
    return model.to(cfg["device"])


def save_checkpoint(path, model, cfg, dataset_manifest_sha256, step, tiny_model,
                    input_protocol, backbone_source_sha256):
    from safetensors.torch import save_model

    path = checkpoint_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    checkpoint = {"format": CHECKPOINT_FORMAT, "serialization": CHECKPOINT_SERIALIZATION,
                  "config": cfg, "step": step, "dataset_manifest_sha256": dataset_manifest_sha256,
                  "labels": list(LABELS), "question": QUESTION, "candidates": list(CANDIDATES),
                  "tiny_model": tiny_model, "input_protocol": input_protocol,
                  "backbone_source_sha256": backbone_source_sha256}
    # save_model preserves shared-storage aliases, including the native tied
    # LM head. Serialization copies tensors to CPU without moving live weights.
    metadata = {"cpv_metadata": json.dumps(checkpoint, ensure_ascii=False,
                                            sort_keys=True, allow_nan=False)}
    try:
        save_model(model, str(temporary), metadata=metadata)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_path(path):
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("Only .safetensors checkpoints are supported; legacy .pt/pickle checkpoints "
                         "are not loaded. Preserve historical runs and create a fresh training run.")
    return path


def read_checkpoint_metadata(path):
    """Read the JSON header without executing legacy pickle deserialization."""
    path = checkpoint_path(path)
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as weights:
        metadata = weights.metadata() or {}
    encoded = metadata.get("cpv_metadata")
    if not isinstance(encoded, str):
        raise ValueError("checkpoint cpv_metadata JSON header is missing")
    try:
        checkpoint = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("checkpoint cpv_metadata JSON header is invalid") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint cpv_metadata JSON header must be an object")
    if checkpoint.get("serialization") != CHECKPOINT_SERIALIZATION:
        raise ValueError("checkpoint serialization format mismatch; expected safetensors-json-v1")
    required = {"format", "config", "step", "dataset_manifest_sha256", "labels", "question",
                "candidates", "tiny_model", "input_protocol", "backbone_source_sha256"}
    if required - checkpoint.keys():
        raise ValueError("checkpoint cpv_metadata JSON header is incomplete")
    if (not isinstance(checkpoint["config"], dict)
            or type(checkpoint["tiny_model"]) is not bool
            or type(checkpoint["step"]) is not int or checkpoint["step"] < 0):
        raise ValueError("checkpoint cpv_metadata config/step/tiny_model types are invalid")
    return checkpoint


def reload_checkpoint(path, args, expected_manifest=None):
    checkpoint = read_checkpoint_metadata(path)
    from safetensors.torch import load_model

    if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("labels") != list(LABELS):
        raise ValueError("checkpoint schema/label order mismatch")
    if checkpoint["question"] != QUESTION or checkpoint["candidates"] != list(CANDIDATES):
        raise ValueError("checkpoint input protocol differs from current code")
    if expected_manifest and checkpoint["dataset_manifest_sha256"] != expected_manifest:
        raise ValueError("checkpoint was trained with a different dataset manifest")
    cfg = dict(checkpoint["config"])
    validate_protocol_config(cfg)
    for key in ("device", "precision", "eval_batch_size"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    validate_execution(cfg)
    args.tiny_model = checkpoint["tiny_model"]
    tokenizer = tokenizer_for(args, cfg)
    if checkpoint.get("input_protocol") != prompt_protocol(tokenizer, cfg):
        raise ValueError("checkpoint tokenizer/prompt/answer-slot protocol differs from current code")
    model = build_model(args, cfg)
    load_model(model, str(path), strict=True, device="cpu")
    return model, cfg, checkpoint


def train(args, cfg, bundle):
    # Keep data/download/package commands independent of framework imports.
    from customer_policy_verifier.training.factory_workflow import train_with_factory

    return train_with_factory(args, cfg, bundle)


def evaluate_checkpoint(args, bundle):
    checkpoint_path(args.checkpoint)
    selection_path = Path(args.checkpoint).parent / "selection.json"
    if not selection_path.exists():
        raise ValueError("selection.json required; finish training and freeze selection before evaluation")
    selection = json.loads(selection_path.read_text())
    if (selection.get("checkpoint") != Path(args.checkpoint).name
            or selection["checkpoint_sha256"] != sha256(args.checkpoint)):
        raise ValueError("checkpoint does not match the selected model")
    report_path = Path(args.output)
    if report_path.exists():
        raise ValueError("evaluation output already exists; choose a fresh --output")
    manifest_hash = sha256(Path(args.data_dir) / "manifest.json")
    model, cfg, checkpoint = reload_checkpoint(args.checkpoint, args, manifest_hash)
    tok = tokenizer_for(args, cfg)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    normalization = cfg.get("input_normalization", "none")
    jev_data.normalization_audit(bundle, normalization)
    rows = [encode_record(r, tok, cfg["max_tokens"], normalization) for r in bundle[args.split]]
    metrics, predictions = evaluate_rows(model, rows, pad, cfg)
    report = {"split": args.split, "metrics": metrics, "checkpoint_sha256": sha256(args.checkpoint),
              "dataset_manifest_sha256": manifest_hash, "selected_step": checkpoint["step"],
              "tiny_model": checkpoint["tiny_model"], "calibration": "none",
              "scope": "TOD-ProcBench Task 2 project-specific group split",
              "input_protocol": checkpoint["input_protocol"],
              "input_audit": input_audit(rows), "predictions": predictions}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report_path, report)
    return {key: value for key, value in report.items() if key != "predictions"}


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    common.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("check", parents=[common], help="offline dataset integrity checks; no ML dependencies")
    pkg = sub.add_parser("package", parents=[common], help="create a portable AutoDL upload; no ML dependencies")
    pkg.add_argument("--output", type=Path, default=DEFAULT_PACKAGE)
    dl = sub.add_parser("download", parents=[common], help="download official Qwen files from ModelScope")
    dl.add_argument("--tokenizer-only", action="store_true")
    audit = sub.add_parser("audit", parents=[common], help="tokenize train and validation; reject over-length inputs")
    audit.add_argument("--output", type=Path, required=True)
    for name in ("smoke", "train"):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--train-set", choices=("original", "mixed"), default="original")
        p.add_argument("--run-dir", type=Path, help="defaults to runs/training/llamafactory-direct-{smoke,<train-set>-v1}")
        p.add_argument("--device", choices=("cpu", "cuda"))
        p.add_argument("--precision", choices=("fp32", "bf16"))
        budget = p.add_mutually_exclusive_group()
        budget.add_argument("--max-steps", type=int, help="override the epoch budget with optimizer updates")
        budget.add_argument("--num-train-epochs", type=int, help="train this many complete epochs")
        for key in ("microbatch_questions", "gradient_accumulation", "eval_batch_size", "eval_every"):
            p.add_argument("--" + key.replace("_", "-"), type=int)
        p.add_argument("--resume-from-checkpoint", type=Path,
                       help="resume the latest complete checkpoints/checkpoint-N in this run")
        if name == "smoke":
            p.add_argument("--samples", type=int, default=4)
            p.add_argument("--tiny-model", action="store_true", help="offline CPU check using a tiny random Qwen; no model-quality claim")
        else:
            p.set_defaults(tiny_model=False)
    ev = sub.add_parser("evaluate", parents=[common], help="explicit evaluation of a frozen selected checkpoint")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--split", choices=("validation", "test"), default="validation")
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--device", choices=("cpu", "cuda"))
    ev.add_argument("--precision", choices=("fp32", "bf16"))
    ev.add_argument("--eval-batch-size", type=int)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ("smoke", "train") and args.run_dir is None:
        name = "llamafactory-direct-smoke" if args.command == "smoke" else "llamafactory-direct-" + args.train_set + "-v1"
        args.run_dir = ROOT / "runs" / "training" / name
    if getattr(args, "tiny_model", False):
        args.device = args.device or "cpu"
        args.precision = args.precision or "fp32"
        args.microbatch_questions = args.microbatch_questions or 1
        args.gradient_accumulation = args.gradient_accumulation or 1
    cfg = config_for(args)
    if args.command == "download":
        result = download(args, cfg)
    else:
        bundle = load_dataset_bundle(args.data_dir)
        normalization = cfg.get("input_normalization", "none")
        normalized_audit = jev_data.normalization_audit(bundle, normalization)
        if args.command == "check":
            result = {"integrity": "passed", "counts": {key: len(bundle[key]) for key in
                      ("train", "train_original", "validation", "test")},
                      "labels": list(LABELS), "model_repo": cfg["model_repo"], "model_revision": cfg["model_revision"]}
            result.update(version=cfg["version"], readout=cfg["readout"], option_letters=cfg["option_letters"])
            result.update(download_source=cfg.get("download_source", "huggingface"),
                          download_revision=cfg["model_revision"])
            result["normalization_audit"] = normalized_audit
        elif args.command == "package":
            result = package_training(args)
        elif args.command == "audit":
            if args.output.exists():
                raise ValueError("audit output already exists; choose a fresh --output")
            tok = tokenizer_for(args, cfg)
            result = {key: input_audit([encode_record(r, tok, cfg["max_tokens"], normalization)
                                       for r in bundle[key]]) for key in jev_data.FILE_NAMES}
            result["input_protocol"] = prompt_protocol(tok, cfg)
            result["normalization_audit"] = normalized_audit
            result["dataset_manifest_sha256"] = sha256(Path(args.data_dir) / "manifest.json")
            result["backbone_source_sha256"] = source_digest(args.backbone)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_json(args.output, result)
        elif args.command == "evaluate":
            result = evaluate_checkpoint(args, bundle)
        else:
            if args.command == "smoke" and args.samples < 1:
                raise ValueError("--samples must be positive")
            result = train(args, cfg, bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
