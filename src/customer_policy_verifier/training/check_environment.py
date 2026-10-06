"""Check pinned packages and a tiny native Qwen training step without downloads."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

from customer_policy_verifier.common.paths import ROOT


def package_versions():
    """Validate the shared requirements, accepting torch's local build suffix."""
    from packaging.version import Version

    expected = {}
    for line in (ROOT / "requirements/training.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, version = line.strip().split("==")
            expected[name] = version
    versions, errors = {}, []
    for name, required in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{name} is missing (expected {required})")
            continue
        versions[name] = actual
        if Version(actual).base_version != required:
            errors.append(f"{name}=={actual}, expected {required}")
    if errors:
        raise ValueError("Training package mismatch: " + "; ".join(errors))
    return versions


def check(device):
    versions = package_versions()
    if sys.version_info[:2] != (3, 12):
        raise ValueError("The selected AutoDL environment requires Python 3.12")
    import torch
    from safetensors.torch import load_model, save_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from customer_policy_verifier.training.jev_model import (
        SemIfDirectModel,
        supervised_loss,
    )

    os_release = platform.freedesktop_os_release() if platform.system() == "Linux" else {}
    report = {"format": "customer-policy-env-v1", "packages": versions,
              "python": platform.python_version(), "platform": platform.platform(),
              "machine": platform.machine(), "os_release": os_release,
              "torch_build_version": str(torch.__version__),
              "torch_cuda_runtime": torch.version.cuda, "device": device,
              "gpu_verified": False, "pretrained_weights_loaded": False}
    if device == "cuda":
        if (platform.system() != "Linux" or platform.machine() != "x86_64"
                or os_release.get("ID") != "ubuntu" or os_release.get("VERSION_ID") != "22.04"):
            raise ValueError("Expected the selected Ubuntu 22.04 / Linux x86_64 AutoDL image")
        # Some vendor/Conda images omit the local '+cu124' metadata suffix;
        # verify the imported build and real runtime rather than that label.
        if str(torch.__version__).split("+")[0] != "2.6.0" or torch.version.cuda != "12.4":
            raise ValueError("Expected a PyTorch 2.6.0 CUDA 12.4 build")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one CUDA GPU, e.g. CUDA_VISIBLE_DEVICES=0")
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        if "4090" not in name or capability != (8, 9):
            raise ValueError(f"Expected RTX 4090 (compute capability 8.9), found {name} / {capability}")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The exposed GPU/runtime does not support bf16")
        driver = None
        try:
            query = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                   check=True, capture_output=True, text=True, timeout=10)
            driver = query.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            pass  # Successful CUDA kernels below also verify driver usability.
        report.update(gpu=name, compute_capability=list(capability), bf16_supported=True,
                      driver_versions=driver, gpu_memory_gb=torch.cuda.get_device_properties(0).total_memory / 1e9)

    torch.manual_seed(20261002)
    cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, tie_word_embeddings=True, max_position_embeddings=32)
    cfg._attn_implementation = "sdpa"
    model = SemIfDirectModel(Qwen3ForCausalLM(cfg)).to(device)
    model.causal_lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    batch = {"input_ids": torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]], device=device),
             "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], device=device),
             "position_ids": torch.tensor([[0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5]], device=device),
             "option_token_ids": torch.tensor([[32, 33], [32, 33]], device=device),
             "cand_mask": torch.ones((2, 2), dtype=torch.bool, device=device),
             "target": torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    precision = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with precision:
        logits = model(batch)["logits"]
    loss = supervised_loss(logits, batch["target"], batch["cand_mask"])
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all()
                                         for p in model.parameters()):
        raise RuntimeError("Tiny Qwen SDPA/checkpoint training produced non-finite or missing gradients")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected_logits = model(batch)["logits"].float()
    # No pickle loading: test the same safe tied-weight representation used by training.
    with tempfile.TemporaryDirectory(prefix="cpv-env-probe-") as folder:
        path = Path(folder) / "probe.safetensors"
        save_model(model, str(path))
        restored = SemIfDirectModel(Qwen3ForCausalLM(cfg)).to(device).eval()
        load_model(restored, str(path), strict=True, device=device)
        with torch.no_grad():
            torch.testing.assert_close(restored(batch)["logits"].float(), expected_logits)
    with redirect_stdout(sys.stderr):
        framework_probe = probe_framework(device)
    report.update(status="passed", gpu_verified=device == "cuda",
                  scope="selected AutoDL GPU environment" if device == "cuda" else "CPU API compatibility; AutoDL GPU/driver not verified",
                  probe={"native_qwen3": True, "logits_to_keep": 1, "sdpa": True,
                         "precision": "bf16" if device == "cuda" else "fp32",
                         "nonreentrant_gradient_checkpoint": True, "adamw_step": True,
                         "finite_loss_and_all_gradients": True, "safetensors_tied_weight_reload": True,
                         **framework_probe})
    return report


def probe_framework(device):
    """Exercise the actual LF trainer and restore its full local checkpoint."""
    from llamafactory.hparams import FinetuningArguments
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from transformers import Qwen3Config, Qwen3ForCausalLM, TrainerCallback

    from customer_policy_verifier.training.factory_trainer import (
        DirectCollator,
        DirectTrainer,
        decision_metrics,
    )
    from customer_policy_verifier.training.factory_workflow import training_arguments

    cfg = json.loads((ROOT / "configs/jev-training.json").read_text())
    cfg.update(device=device, precision="bf16" if device == "cuda" else "fp32", max_steps=2,
               eval_every=1, gradient_accumulation=2, warmup_steps=0, logging_every=1,
               report_to=["tensorboard"])
    native_config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                               head_dim=8, tie_word_embeddings=True, max_position_embeddings=32)
    native_config._attn_implementation = "sdpa"
    rows = [{"input_ids": [1, 2, 3, 4], "option_token_ids": [32, 33], "target": [1.0, 0.0]},
            {"input_ids": [1, 2, 5, 6], "option_token_ids": [32, 33], "target": [0.0, 1.0]}]

    class StopAfterOne(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True
            return control

    restored_optimizer_steps = []

    class CaptureResume(TrainerCallback):
        def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
            if state.global_step == 1:
                restored_optimizer_steps.extend(float(v["step"]) for v in optimizer.state.values())

    with tempfile.TemporaryDirectory(prefix="cpv-factory-probe-") as folder:
        def create(callback):
            model = Qwen3ForCausalLM(native_config)
            model.config.use_cache = False
            return DirectTrainer(model=model, args=training_arguments(cfg, folder, smoke=True),
                                 finetuning_args=FinetuningArguments(stage="sft", finetuning_type="full"),
                                 tokenizer=None, processor=None, data_collator=DirectCollator(0),
                                 train_dataset=rows, eval_dataset=rows, compute_metrics=decision_metrics,
                                 callbacks=[callback])
        first = create(StopAfterOne())
        first.train()
        del first
        resumed = create(CaptureResume())
        resumed.train(resume_from_checkpoint=str(Path(folder) / "checkpoints/checkpoint-1"))
        if resumed.state.global_step != 2 or not restored_optimizer_steps or any(
                step != 1 for step in restored_optimizer_steps):
            raise RuntimeError("LLaMA-Factory checkpoint did not restore optimizer step one")
        del resumed
        events = EventAccumulator(str(Path(folder) / "tensorboard")).Reload()
        if not {"train/loss", "train/learning_rate", "eval/cross_entropy"} <= set(events.Tags()["scalars"]):
            raise RuntimeError("TensorBoard did not save the expected training/validation scalars")
        if 2 not in [event.step for event in events.Scalars("train/loss")]:
            raise RuntimeError("TensorBoard did not save the resumed training update")
    return {"llamafactory_direct_trainer": True, "framework_gradient_accumulation": True,
            "optimizer_scheduler_rng_resume": True, "tensorboard_scalars_saved": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        raise ValueError("Environment report exists; select a fresh --output")
    result = check(args.device)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return result


if __name__ == "__main__":
    main()
