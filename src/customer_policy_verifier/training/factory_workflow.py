"""Full-parameter Direct training using LLaMA-Factory's training lifecycle."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch
from llamafactory.hparams import FinetuningArguments, TrainingArguments
from transformers import TrainerCallback, TrainerState, set_seed

from customer_policy_verifier.training import (
    backbone_download,
    factory_trainer,
    jev_data,
    jev_model,
)
from customer_policy_verifier.training import train_jev as io
from customer_policy_verifier.training.factory_trainer import (
    DirectCollator,
    DirectTrainer,
    decision_metrics,
)

FRAMEWORK_PACKAGES = ("torch", "transformers", "llamafactory", "accelerate", "peft", "trl", "tensorboard")


def framework_versions():
    return {name: importlib.metadata.version(name) for name in FRAMEWORK_PACKAGES}


def training_arguments(cfg, run_dir, *, smoke=False):
    return TrainingArguments(
        output_dir=str(Path(run_dir) / "checkpoints"),
        do_train=True, do_eval=True, use_cpu=cfg["device"] == "cpu",
        bf16=cfg["precision"] == "bf16", fp16=False,
        per_device_train_batch_size=cfg["microbatch_questions"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation"],
        max_steps=cfg["max_steps"], num_train_epochs=cfg.get("num_train_epochs", 3),
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"], optim=cfg["optimizer"],
        lr_scheduler_type=cfg["lr_scheduler"], warmup_steps=cfg.get("warmup_steps", 0),
        warmup_ratio=cfg.get("warmup_ratio", 0.0),
        max_grad_norm=cfg["grad_clip"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy=cfg.get("eval_strategy", "steps"), eval_steps=cfg["eval_every"],
        save_strategy=cfg.get("eval_strategy", "steps"), save_steps=cfg["eval_every"],
        save_total_limit=cfg["save_total_limit"], save_safetensors=True,
        logging_strategy="steps", logging_steps=1 if smoke else cfg["logging_every"],
        report_to=cfg.get("report_to", ["tensorboard"]),
        logging_dir=str(Path(run_dir) / "tensorboard"),
        push_to_hub=False, predict_with_generate=False,
        remove_unused_columns=False, label_names=["target"],
        dataloader_num_workers=0, dataloader_pin_memory=cfg["device"] == "cuda",
        seed=cfg["seed"], data_seed=cfg["seed"], disable_tqdm=True,
        # Project selection also includes step zero, unlike HF's default
        # load_best_model_at_end selection among training checkpoints only.
        load_best_model_at_end=False,
    )


def training_plan(trainer):
    """Use the installed Trainer's own budget calculation, including short windows."""
    values = trainer.set_initial_training_values(
        trainer.args, trainer.get_train_dataloader(), trainer.get_total_train_batch_size(trainer.args),
    )
    _, updates_per_epoch, questions, _, epoch_based, microbatches, optimizer_steps = values
    return {"mode": "epochs" if epoch_based else "steps",
            "num_train_epochs": trainer.args.num_train_epochs if epoch_based else None,
            "optimizer_steps": optimizer_steps, "updates_per_epoch": updates_per_epoch,
            "questions": questions, "microbatches_per_epoch": microbatches,
            "warmup_steps": trainer.args.get_warmup_steps(optimizer_steps),
            "warmup_ratio": trainer.args.warmup_ratio}


def validation_metrics(metrics):
    return {key: metrics["eval_" + key] for key in
            ("n", "accuracy", "macro_f1", "violating_recall", "compliant_false_positive_rate",
             "confusion_matrix", "cross_entropy", "brier_sum")}


class SelectionCallback(TrainerCallback):
    def __init__(self, run_dir, cfg, manifest_hash, protocol, source_hash, tiny_model,
                 best_loss, best_step=0, elapsed_seconds=0.0):
        self.run_dir = Path(run_dir)
        self.cfg, self.manifest_hash = cfg, manifest_hash
        self.protocol, self.source_hash = protocol, source_hash
        self.tiny_model = tiny_model
        self.best_loss, self.best_step = best_loss, best_step
        self.elapsed_seconds = elapsed_seconds
        self.started = time.monotonic()

    def elapsed(self):
        return self.elapsed_seconds + time.monotonic() - self.started

    def on_step_end(self, args, state, control, **kwargs):
        # Evaluate and save the final update even when max_steps is not a
        # multiple of eval_steps, so the latest complete checkpoint can resume.
        if state.global_step == state.max_steps:
            control.should_log = True
            if args.eval_strategy.value == "steps":
                control.should_evaluate = control.should_save = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with (self.run_dir / "events.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": state.global_step, "elapsed_seconds": self.elapsed(),
                                         **logs}, allow_nan=False) + "\n")

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        value = metrics["eval_cross_entropy"]
        if not math.isfinite(value):
            raise RuntimeError("non-finite validation cross entropy")
        if value < self.best_loss:
            self.best_loss, self.best_step = value, state.global_step
            io.save_checkpoint(self.run_dir / io.CHECKPOINT_NAME, jev_model.SemIfDirectModel(model),
                               self.cfg, self.manifest_hash, self.best_step, self.tiny_model,
                               self.protocol, self.source_hash)

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        # save_checkpoint replaces best atomically. A hard link preserves this
        # selection if a later evaluation improves best but its checkpoint
        # save is interrupted, without copying another full model on disk.
        snapshot = checkpoint / "cpv_best.safetensors"
        snapshot.unlink(missing_ok=True)
        os.link(self.run_dir / io.CHECKPOINT_NAME, snapshot)
        io.save_json(checkpoint / "cpv_resume.json", {
            "format": "cpv-llamafactory-resume-v1", "step": state.global_step,
            "planned_steps": state.max_steps,
            "best_step": self.best_step, "best_loss": self.best_loss,
            "best_checkpoint_sha256": io.sha256(self.run_dir / io.CHECKPOINT_NAME),
            "elapsed_seconds": self.elapsed(),
            "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1e9
            if self.cfg["device"] == "cuda" else None,
        })


def validate_resume(args, cfg, manifest_hash, protocol, script_hashes):
    checkpoint = Path(args.resume_from_checkpoint).resolve()
    run_dir = Path(args.run_dir).resolve()
    if checkpoint.parent != run_dir / "checkpoints":
        raise ValueError("resume checkpoint must belong to this run's checkpoints directory")
    required = ("model.safetensors", "config.json", "trainer_state.json", "optimizer.pt",
                "scheduler.pt", "rng_state.pth", "cpv_resume.json", "cpv_best.safetensors")
    if any(not (checkpoint / name).is_file() for name in required):
        raise ValueError("resume requires a complete LLaMA-Factory checkpoint, including optimizer/RNG state")
    complete = [p for p in checkpoint.parent.glob("checkpoint-*")
                if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()
                and (p / "cpv_resume.json").is_file()]
    if checkpoint != max(complete, key=lambda p: int(p.name.split("-")[-1])):
        raise ValueError("resume from the latest complete checkpoint to preserve run selection history")
    run = json.loads((run_dir / "run.json").read_text())
    checks = {"config": cfg, "train_set": args.train_set,
              "smoke": args.command == "smoke", "tiny_model": args.tiny_model,
              "samples": args.samples if args.command == "smoke" else None,
              "dataset_manifest_sha256": manifest_hash, "input_protocol": protocol,
              "script_sha256": script_hashes}
    if any(run.get(key) != value for key, value in checks.items()):
        raise ValueError("resume configuration, data, prompt, or training code differs from the original run")
    if run["runtime"]["framework_packages"] != framework_versions():
        raise ValueError("resume framework package versions differ from the original run")
    restored = json.loads((checkpoint / "cpv_resume.json").read_text())
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if (restored.get("format") != "cpv-llamafactory-resume-v1"
            or restored.get("step") != state["global_step"]
            or restored.get("planned_steps") != state["max_steps"]
            or restored.get("planned_steps") != run["training_plan"]["optimizer_steps"]
            or restored["step"] != int(checkpoint.name.split("-")[-1])
            or restored["best_checkpoint_sha256"] != io.sha256(checkpoint / "cpv_best.safetensors")):
        raise ValueError("resume state or selected checkpoint changed")
    if restored["step"] > restored["planned_steps"]:
        raise ValueError("resume step exceeds the planned training budget")
    if restored["step"] == restored["planned_steps"] and (run_dir / "selection.json").exists():
        raise ValueError("training already complete; evaluate the selected model instead")
    # A newer, incomplete save may have changed the mutable selection file.
    # Restore the exact selection from the complete checkpoint being resumed.
    best = run_dir / io.CHECKPOINT_NAME
    if not best.exists() or io.sha256(best) != restored["best_checkpoint_sha256"]:
        temporary = best.with_suffix(".resume.tmp")
        temporary.unlink(missing_ok=True)
        os.link(checkpoint / "cpv_best.safetensors", temporary)
        temporary.replace(best)
    return restored


def train_with_factory(args, cfg, bundle):
    if cfg.get("training_backend") != "llamafactory" or cfg.get("finetuning_type") != "full":
        raise ValueError("training requires the LLaMA-Factory full-parameter configuration")
    smoke = args.command == "smoke"
    run_dir = Path(args.run_dir)
    resume = getattr(args, "resume_from_checkpoint", None)
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("run directory is not empty; select a fresh --run-dir or resume its latest checkpoint")
    train_records = bundle["train_original" if args.train_set == "original" else "train"]
    validation_records = bundle["validation"]
    if smoke:
        train_records, validation_records = train_records[:args.samples], validation_records[:args.samples]
        cfg = dict(cfg, max_steps=min(cfg["max_steps"], 2) if cfg["max_steps"] > 0 else 2,
                   eval_every=1, eval_strategy="steps")
    tok = io.tokenizer_for(args, cfg)
    normalization = cfg.get("input_normalization", "none")
    normalized_audit = jev_data.normalization_audit(bundle, normalization)
    protocol = io.prompt_protocol(tok, cfg)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise ValueError("tokenizer has no padding or EOS token")
    train_rows = [jev_data.encode_record(r, tok, cfg["max_tokens"], normalization) for r in train_records]
    validation_rows = [jev_data.encode_record(r, tok, cfg["max_tokens"], normalization) for r in validation_records]
    manifest_hash = io.sha256(Path(args.data_dir) / "manifest.json")
    script_hashes = {p.resolve().relative_to(io.ROOT).as_posix(): io.sha256(p) for p in
                     (Path(io.__file__), Path(__file__), Path(factory_trainer.__file__),
                      Path(jev_data.__file__), Path(jev_model.__file__), Path(backbone_download.__file__))}
    restored = validate_resume(args, cfg, manifest_hash, protocol, script_hashes) if resume else None
    set_seed(cfg["seed"])
    wrapper = io.build_model(args, cfg)
    model = wrapper.causal_lm
    source_hash = io.source_digest(args.backbone) if not args.tiny_model else None
    trainer = DirectTrainer(
        model=model, args=training_arguments(cfg, run_dir, smoke=smoke),
        finetuning_args=FinetuningArguments(stage="sft", finetuning_type="full"),
        processor=None, tokenizer=None if args.tiny_model else tok,
        data_collator=DirectCollator(pad), train_dataset=train_rows, eval_dataset=validation_rows,
        compute_metrics=lambda result: decision_metrics(result, cfg["violating_threshold"]),
    )
    if trainer.args.world_size != 1:
        raise ValueError("this workflow currently supports one CPU process or one RTX 4090")
    plan = training_plan(trainer)
    if resume and restored["planned_steps"] != plan["optimizer_steps"]:
        raise ValueError("resume training budget differs from the current framework/data loader")
    if cfg["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=True)
        io.save_json(run_dir / "run.json", {
            "config": cfg, "train_set": args.train_set, "smoke": smoke, "tiny_model": args.tiny_model,
            "samples": args.samples if smoke else None,
            "runtime": {"python": sys.version, "platform": platform.platform(),
                        "framework_packages": framework_versions(), "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name() if cfg["device"] == "cuda" else None},
            "dataset_manifest_sha256": manifest_hash, "backbone_source_sha256": source_hash,
            "script_sha256": script_hashes, "input_protocol": protocol,
            "normalization_audit": normalized_audit,
            "parameters": {"all": sum(p.numel() for p in model.parameters()),
                           "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
            "input_audit": {"train": io.input_audit(train_rows), "validation": io.input_audit(validation_rows)},
            "framework_arguments": trainer.args.to_dict(),
            "training_plan": plan,
        })
        initial = validation_metrics(trainer.evaluate())
        io.save_json(run_dir / "initial_validation.json", initial)
        io.save_checkpoint(run_dir / io.CHECKPOINT_NAME, wrapper, cfg, manifest_hash, 0,
                           args.tiny_model, protocol, source_hash)
        restored = {"best_loss": initial["cross_entropy"], "best_step": 0, "elapsed_seconds": 0.0}
    callback = SelectionCallback(run_dir, cfg, manifest_hash, protocol, source_hash, args.tiny_model,
                                 restored["best_loss"], restored["best_step"], restored["elapsed_seconds"])
    trainer.add_callback(callback)
    if resume and restored["step"] == plan["optimizer_steps"]:
        # A final complete training checkpoint may exist even if export or
        # reload validation was interrupted. Finish those without more updates.
        trainer.state = TrainerState.load_from_json(Path(resume) / "trainer_state.json")
    else:
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
    if trainer.state.global_step != plan["optimizer_steps"]:
        raise RuntimeError("training stopped before max_steps; resume the latest complete checkpoint")
    if plan["mode"] == "epochs" and abs((trainer.state.epoch or 0) - plan["num_train_epochs"]) > 1e-6:
        raise RuntimeError("training ended without completing the requested epochs")
    selection = {
        "best_step": callback.best_step, "validation_cross_entropy": callback.best_loss,
        "checkpoint": io.CHECKPOINT_NAME, "checkpoint_sha256": io.sha256(run_dir / io.CHECKPOINT_NAME),
        "training_backend": "llamafactory", "test_evaluated": False,
        "threshold": cfg["violating_threshold"], "training_seconds": callback.elapsed(),
        "peak_cuda_memory_gb": max(restored.get("peak_cuda_memory_gb") or 0,
                                   torch.cuda.max_memory_allocated() / 1e9)
        if cfg["device"] == "cuda" else None,
        "completed_steps": trainer.state.global_step,
        "completed_epochs": trainer.state.epoch,
        "training_plan": plan,
        "tensorboard_dir": "tensorboard" if "tensorboard" in trainer.args.report_to else None,
    }
    # Free optimizer and live CUDA tensors before constructing a fresh model.
    wrapper.to("cpu")
    del trainer, model, wrapper
    gc.collect()
    if cfg["device"] == "cuda":
        torch.cuda.empty_cache()
    reloaded, restored_cfg, checkpoint = io.reload_checkpoint(run_dir / io.CHECKPOINT_NAME, args, manifest_hash)
    restored_metrics, _ = io.evaluate_rows(reloaded, validation_rows, pad, restored_cfg)
    if abs(restored_metrics["cross_entropy"] - callback.best_loss) > 1e-5:
        raise RuntimeError("checkpoint reload changed validation predictions")
    io.save_json(run_dir / "reload_check.json", {"passed": True, "validation": restored_metrics})
    # Export the selected weights, not the final update's weights. Standard HF
    # model files can be loaded offline without this project's wrapper.
    export = run_dir / "export"
    reloaded.causal_lm.save_pretrained(export, safe_serialization=True)
    if not args.tiny_model:
        tok.save_pretrained(export)
    io.save_json(export / "cpv_model.json", checkpoint)
    selection["export"] = "export"
    io.save_json(run_dir / "selection.json", selection)
    return selection
