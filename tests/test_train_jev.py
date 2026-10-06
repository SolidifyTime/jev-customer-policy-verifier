"""Offline training lifecycle checks with tiny random Qwen3 weights only."""

import contextlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tests"))

from test_jev_data import fixture_records, write_bundle

from customer_policy_verifier.training import train_jev as training

DEPENDENCIES = (importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("safetensors") is not None
                and importlib.util.find_spec("llamafactory") is not None)
if DEPENDENCIES:
    import torch
    from safetensors import SafetensorError, safe_open
    from safetensors.torch import load_file, load_model, save_file

    from customer_policy_verifier.training import factory_workflow
    from customer_policy_verifier.training.factory_trainer import (
        decision_metrics,
    )


class JevPackageTests(unittest.TestCase):
    def test_new_config_and_ambiguous_warmup_validation(self):
        config = PROJECT / "configs/jev-training-case-neutral-1epoch.json"
        cfg = training.config_for(training.parser().parse_args(["check", "--config", str(config)]))
        self.assertEqual((cfg["num_train_epochs"], cfg["max_steps"], cfg["input_normalization"]),
                         (1, -1, "lowercase"))
        self.assertEqual((cfg["warmup_steps"], cfg["warmup_ratio"]), (0, 0.1))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for change in ({"warmup_steps": 20}, {"warmup_ratio": float("nan")},
                           {"warmup_ratio": True}, {"warmup_steps": 1.5}):
                with self.subTest(change=change):
                    path.write_text(json.dumps(dict(cfg, **change)))
                    with self.assertRaisesRegex(ValueError, "warmup"):
                        training.config_for(training.parser().parse_args(["check", "--config", str(path)]))

    def test_portable_package_contains_current_sources_and_runs_cli_check(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            data_dir = directory / "fixed-data"
            data_dir.mkdir()
            write_bundle(data_dir, fixture_records())
            archive_path = directory / "upload.tar.gz"
            with contextlib.redirect_stdout(io.StringIO()):
                package = training.main(["package", "--data-dir", str(data_dir), "--output", str(archive_path)])
            packaged = set(package["files"])
            source_files = {path.relative_to(PROJECT).as_posix() for path in
                            (PROJECT / "src" / "customer_policy_verifier").rglob("*.py")}
            self.assertTrue(source_files <= packaged)
            required = {"pyproject.toml", "scripts/cpv.py", "requirements/training.txt",
                        "requirements/autodl-cu124.txt",
                        "docs/guides/jev-training.md", "configs/jev-training.json",
                        "configs/layout-migration.json", "LICENSES/semif-openjev-MIT.txt",
                        "configs/archive/agentjev-training-v1.json", "docs/archive/agentjev-training-v1.md"}
            required.update("data/training/task2-v1/" + name for name in
                            ("manifest.json", "train.jsonl", "train_original.jsonl", "validation.jsonl", "test.jsonl"))
            self.assertTrue(required <= packaged)
            self.assertFalse(any((".env" in name and name != ".env.example") or "__pycache__" in name
                                 for name in packaged))
            self.assertFalse(any(name.startswith(("models/", "data/source/", "data/derived/", "data/archive/"))
                                 for name in packaged))
            self.assertTrue({name for name in packaged if name.startswith("runs/")}
                            <= {"runs/README.md", "runs/catalog.json"})
            extracted = directory / "extracted"
            extracted.mkdir()
            with tarfile.open(archive_path, "r:gz") as archive:
                self.assertEqual({member.name for member in archive.getmembers()},
                                 {"customer-policy-verifier/" + name for name in packaged})
                archive.extractall(extracted, filter="data")
            project = extracted / "customer-policy-verifier"
            for name, digest in package["files"].items():
                self.assertEqual(training.sha256(project / name), digest)
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            environment.pop("PYTHONPATH", None)
            environment.pop("CPV_PROJECT_ROOT", None)
            result = subprocess.run([sys.executable, "scripts/cpv.py", "train", "check"],
                                    cwd=project, env=environment, text=True, capture_output=True, check=True)
            checked = json.loads(result.stdout)
            self.assertEqual(checked["integrity"], "passed")
            self.assertEqual(checked["counts"], {"train": 3, "train_original": 2, "validation": 1, "test": 1})
            with self.assertRaisesRegex(ValueError, "package output already exists"):
                training.main(["package", "--data-dir", str(data_dir), "--output", str(archive_path)])

    def test_package_output_default_uses_new_package_directory(self):
        arguments = training.parser().parse_args(["package"])
        self.assertEqual(arguments.output, PROJECT / "runs/packages/autodl/customer-policy-verifier-training.tar.gz")


@unittest.skipUnless(DEPENDENCIES, "torch, transformers and safetensors are required for tiny-model integration tests")
class JevTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "portable-data"
        self.data.mkdir()
        write_bundle(self.data, fixture_records())
        cfg = json.loads(training.DEFAULT_CONFIG.read_text())
        cfg.update(max_steps=2, max_tokens=256, microbatch_questions=2,
                   gradient_accumulation=2, eval_batch_size=2, eval_every=1,
                   warmup_steps=0, learning_rate=0.001,
                   device="cpu", precision="fp32")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(cfg))
        self.run = self.root / "smoke-run"
        self.common = ["--config", str(self.config), "--data-dir", str(self.data)]
        self.smoke = ["smoke", *self.common, "--run-dir", str(self.run),
                      "--tiny-model", "--samples", "3", "--train-set", "mixed",
                      "--microbatch-questions", "2", "--gradient-accumulation", "2"]

    def quiet_main(self, argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return training.main(argv)

    def fixture_checkpoint(self):
        args = training.parser().parse_args(self.smoke)
        cfg = json.loads(self.config.read_text())
        model = training.build_model(args, cfg)
        path = self.root / "fixture.safetensors"
        training.save_checkpoint(path, model, cfg, training.sha256(self.data / "manifest.json"),
                                 0, True, training.prompt_protocol(training.TinyTokenizer()), None)
        with safe_open(str(path), framework="pt", device="cpu") as weights:
            header = weights.metadata()
        return args, path, header

    def test_full_parameter_smoke_saves_reloads_and_evaluates_a_portable_bundle(self):
        instances, initial_states = [], []
        original_builder = training.build_model

        def capture_model(args, cfg):
            model = original_builder(args, cfg)
            instances.append(model)
            initial_states.append({name: value.detach().clone()
                                   for name, value in model.state_dict().items()})
            return model

        adamw = torch.optim.AdamW
        with patch.object(training, "build_model", side_effect=capture_model), \
                patch("torch.optim.AdamW", wraps=adamw) as optimizer_constructor:
            selection = self.quiet_main(self.smoke)

        self.assertEqual(len(instances), 2, "training must reload the selected checkpoint")
        trained = instances[0]
        groups = optimizer_constructor.call_args.args[0]
        self.assertEqual(len(groups), 2, "framework separates decay and no-decay parameters")
        optimized = [parameter for group in groups for parameter in group["params"]]
        expected = list(trained.parameters())
        self.assertEqual({id(p) for p in optimized}, {id(p) for p in expected})
        self.assertEqual(len(optimized), len(expected), "each parameter must appear exactly once")
        self.assertTrue(all(p.requires_grad for p in expected))
        changed = [name for name, value in trained.state_dict().items()
                   if not torch.equal(value.cpu(), initial_states[0][name].cpu())]
        self.assertTrue(any(name.startswith("causal_lm.model.layers.") for name in changed))
        self.assertIn("causal_lm.lm_head.weight", changed)
        self.assertIs(trained.causal_lm.lm_head.weight, trained.causal_lm.model.embed_tokens.weight)

        events = [json.loads(line) for line in (self.run / "events.jsonl").read_text().splitlines()]
        losses = [event for event in events if "loss" in event]
        self.assertEqual([event["step"] for event in losses], [1, 2])
        self.assertTrue(all(math.isfinite(event["loss"]) for event in losses))
        self.assertFalse(selection["test_evaluated"])
        self.assertIn(selection["best_step"], (0, 1, 2))
        self.assertEqual(selection["checkpoint"], "best.safetensors")
        self.assertEqual(selection["checkpoint_sha256"], training.sha256(self.run / "best.safetensors"))
        self.assertFalse((self.run / "best.safetensors.tmp").exists())
        self.assertFalse((self.run / "best.pt").exists())
        run = json.loads((self.run / "run.json").read_text())
        self.assertEqual(set(run["script_sha256"]), {
            "src/customer_policy_verifier/training/jev_data.py",
            "src/customer_policy_verifier/training/jev_model.py",
            "src/customer_policy_verifier/training/train_jev.py",
            "src/customer_policy_verifier/training/factory_trainer.py",
            "src/customer_policy_verifier/training/factory_workflow.py",
            "src/customer_policy_verifier/training/backbone_download.py",
        })
        for name, digest in run["script_sha256"].items():
            self.assertEqual(training.sha256(PROJECT / name), digest)
        self.assertEqual(run["parameters"]["all"], run["parameters"]["trainable"])
        self.assertEqual(set(run["input_audit"]), {"train", "validation"})
        self.assertEqual(run["input_audit"]["train"]["questions"], 3)
        self.assertEqual(run["input_audit"]["train"]["prompts"], 3)
        self.assertEqual(run["input_audit"]["train"]["truncated"], 0)
        self.assertEqual(run["input_protocol"]["option_letters"], ["A", "B"])
        self.assertEqual(run["config"]["version"], training.CHECKPOINT_FORMAT)
        reload_check = json.loads((self.run / "reload_check.json").read_text())
        self.assertTrue(reload_check["passed"])
        self.assertAlmostEqual(reload_check["validation"]["cross_entropy"],
                               selection["validation_cross_entropy"], places=5)

        for split in ("validation", "test"):
            output = self.root / f"{split}.json"
            report = self.quiet_main(["evaluate", *self.common, "--checkpoint", str(self.run / "best.safetensors"),
                                      "--split", split, "--output", str(output)])
            self.assertEqual(report["split"], split)
            self.assertEqual(report["metrics"]["n"], 1)
            self.assertEqual(report["input_audit"]["truncated"], 0)
            self.assertTrue(report["tiny_model"])
            full_report = json.loads(output.read_text())
            probabilities = full_report["predictions"][0]["probabilities"]
            self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities))
            self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
            with self.assertRaisesRegex(ValueError, "output already exists"):
                self.quiet_main(["evaluate", *self.common, "--checkpoint", str(self.run / "best.safetensors"),
                                 "--split", split, "--output", str(output)])

        with self.assertRaisesRegex(ValueError, "run directory is not empty"):
            self.quiet_main(self.smoke)
        # A legacy head checkpoint or a new checkpoint with different answer
        # slots must fail before constructing or loading a model.
        checkpoint_args = training.parser().parse_args([
            "evaluate", *self.common, "--checkpoint", str(self.run / "best.safetensors"),
            "--output", str(self.root / "unused.json"),
        ])
        saved = training.read_checkpoint_metadata(self.run / "best.safetensors")
        self.assertEqual(saved["input_protocol"], run["input_protocol"])
        self.assertEqual(saved["serialization"], "safetensors-json-v1")
        self.assertNotIn("state_dict", saved)
        self.assertNotIn("optimizer", saved)
        with safe_open(str(self.run / "best.safetensors"), framework="pt", device="cpu") as checkpoint:
            header = checkpoint.metadata()
        bad = self.root / "incompatible.safetensors"
        tensors = load_file(self.run / "best.safetensors")
        save_file(tensors, str(bad), metadata=dict(
            header, cpv_metadata=json.dumps(dict(saved, format="customer-policy-jev-v1"))))
        with self.assertRaisesRegex(ValueError, "schema/label order"):
            training.reload_checkpoint(bad, checkpoint_args)
        altered_protocol = dict(saved["input_protocol"], option_token_ids=[3, 2])
        save_file(tensors, str(bad), metadata=dict(
            header, cpv_metadata=json.dumps(dict(saved, input_protocol=altered_protocol))))
        with self.assertRaisesRegex(ValueError, "tokenizer/prompt/answer-slot protocol"):
            training.reload_checkpoint(bad, checkpoint_args)
        with (self.run / "best.safetensors").open("ab") as checkpoint:
            checkpoint.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "does not match the selected model"):
            self.quiet_main(["evaluate", *self.common, "--checkpoint", str(self.run / "best.safetensors"),
                             "--output", str(self.root / "tampered-report.json")])

    def test_initial_checkpoint_is_retained_if_validation_never_improves(self):
        initial_state = {}
        original_builder = training.build_model
        original_evaluate = training.evaluate_rows
        losses = iter((0.6, 0.7, 0.8))

        def capture_initial(args, cfg):
            model = original_builder(args, cfg)
            if not initial_state:
                initial_state.update({name: value.detach().clone()
                                      for name, value in model.state_dict().items()})
            return model

        def worsening_validation(result, threshold):
            return dict(decision_metrics(result, threshold), cross_entropy=next(losses))

        def restored_validation(model, rows, pad, cfg):
            metrics, predictions = original_evaluate(model, rows, pad, cfg)
            return dict(metrics, cross_entropy=0.6), predictions

        with patch.object(training, "build_model", side_effect=capture_initial), \
                patch.object(factory_workflow, "decision_metrics", worsening_validation), \
                patch.object(training, "evaluate_rows", side_effect=restored_validation):
            selection = self.quiet_main(self.smoke)
        self.assertEqual(selection["best_step"], 0)
        checkpoint_path = self.run / "best.safetensors"
        checkpoint = training.read_checkpoint_metadata(checkpoint_path)
        self.assertEqual(checkpoint["step"], 0)
        restored = original_builder(
            training.parser().parse_args(self.smoke), json.loads(self.config.read_text()))
        load_model(restored, str(checkpoint_path), strict=True, device="cpu")
        self.assertIs(restored.causal_lm.lm_head.weight, restored.causal_lm.model.embed_tokens.weight)
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as weights:
            names = set(weights.keys())
        tied_names = {"causal_lm.lm_head.weight", "causal_lm.model.embed_tokens.weight"}
        self.assertEqual(len(names & tied_names), 1, "tied weights must be stored only once")
        self.assertEqual(len(names), len(initial_state) - 1)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)
        self.assertTrue(json.loads((self.run / "reload_check.json").read_text())["passed"])

    def test_legacy_pickle_and_disguised_files_are_rejected_without_loading_a_model(self):
        legacy = self.root / "best.pt"
        legacy.write_bytes(b"legacy pickle fixture; never deserialize")
        args = training.parser().parse_args(self.smoke)
        with patch.object(training, "build_model", side_effect=AssertionError("unexpected model construction")):
            with self.assertRaisesRegex(ValueError, r"legacy \.pt/pickle checkpoints are not loaded"):
                training.read_checkpoint_metadata(legacy)
            with self.assertRaisesRegex(ValueError, r"legacy \.pt/pickle checkpoints are not loaded"):
                training.reload_checkpoint(legacy, args)
            with self.assertRaisesRegex(ValueError, r"legacy \.pt/pickle checkpoints are not loaded"):
                self.quiet_main(["evaluate", *self.common, "--checkpoint", str(legacy),
                                 "--output", str(self.root / "never-created.json")])
            disguised = self.root / "legacy.safetensors"
            disguised.write_bytes(legacy.read_bytes())
            with self.assertRaises(SafetensorError):
                training.reload_checkpoint(disguised, args)
        self.assertFalse((self.root / "never-created.json").exists())

    def test_invalid_checkpoint_metadata_is_rejected_before_model_construction(self):
        args, path, _header = self.fixture_checkpoint()
        saved = training.read_checkpoint_metadata(path)
        tensors = load_file(path)
        incomplete = dict(saved)
        del incomplete["step"]
        variants = (
            ({}, "JSON header is missing"),
            ({"cpv_metadata": "{"}, "JSON header is invalid"),
            ({"cpv_metadata": "[]"}, "JSON header must be an object"),
            ({"cpv_metadata": json.dumps(dict(saved, serialization="legacy-pickle"))},
             "serialization format mismatch"),
            ({"cpv_metadata": json.dumps(incomplete)}, "JSON header is incomplete"),
            ({"cpv_metadata": json.dumps(dict(saved, tiny_model=1))}, "types are invalid"),
            ({"cpv_metadata": json.dumps(dict(saved, labels=["violating", "compliant"]))},
             "schema/label order mismatch"),
        )
        bad = self.root / "invalid-metadata.safetensors"
        with patch.object(training, "build_model", side_effect=AssertionError("unexpected model construction")):
            for metadata, expected in variants:
                with self.subTest(expected=expected):
                    save_file(tensors, str(bad), metadata=metadata)
                    with self.assertRaisesRegex(ValueError, expected):
                        training.reload_checkpoint(bad, args)

    def test_checkpoint_loading_strictly_rejects_missing_extra_shape_and_dtype_changes(self):
        args, path, header = self.fixture_checkpoint()
        tensors = load_file(path)
        name = "causal_lm.model.norm.weight"
        missing = dict(tensors)
        del missing[name]
        variants = (
            (missing, "Missing key"),
            (dict(tensors, unexpected_fixture_weight=torch.zeros(1)), "Unexpected key"),
            (dict(tensors, **{name: tensors[name][:-1]}), "Invalid key"),
            (dict(tensors, **{name: tensors[name].double()}), "Invalid key"),
        )
        bad = self.root / "invalid-weights.safetensors"
        for state, expected in variants:
            with self.subTest(expected=expected):
                save_file(state, str(bad), metadata=header)
                with self.assertRaisesRegex(RuntimeError, expected):
                    training.reload_checkpoint(bad, args)

    def test_regular_loader_trains_a_local_random_qwen_checkpoint_without_network(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

        backbone = self.root / "local-tiny-backbone"
        backbone.mkdir()
        corpus = training.QUESTION + " " + " ".join(training.CANDIDATES) + " A B"
        vocabulary = {"[PAD]": 0, "[UNK]": 1}
        vocabulary.update({word: index + 2 for index, word in enumerate(sorted(set(corpus.split())))})
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]")
        fast.chat_template = (
            "{% for message in messages %}{{ message['role'] + '\\n' + message['content'] + '\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}assistant\n{% endif %}"
        )
        fast.save_pretrained(backbone)
        base = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=256, tie_word_embeddings=True,
        ))
        base.save_pretrained(backbone)
        cfg = json.loads(self.config.read_text())
        cfg.update(model_repo="offline-fixture/tiny-qwen3", model_revision="fixture-version",
                   max_steps=1)
        self.config.write_text(json.dumps(cfg))
        training.save_json(backbone / "backbone-source.json", {
            "repo_id": cfg["model_repo"], "revision": cfg["model_revision"], "has_weights": True,
            "files": {path.name: training.sha256(path) for path in backbone.iterdir() if path.is_file()},
        })
        with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected model download")):
            selection = self.quiet_main([
                "train", *self.common, "--backbone", str(backbone), "--run-dir", str(self.run),
                "--train-set", "original", "--max-steps", "1",
            ])
            report = self.quiet_main([
                "evaluate", *self.common, "--backbone", str(backbone),
                "--checkpoint", str(self.run / "best.safetensors"), "--output", str(self.root / "regular-evaluation.json"),
            ])
        self.assertFalse(report["tiny_model"], "this checks the ordinary pretrained-loading path")
        self.assertEqual(report["metrics"]["n"], 1)
        self.assertFalse(selection["test_evaluated"])
        self.assertTrue(json.loads((self.run / "reload_check.json").read_text())["passed"])

    def test_zero_training_steps_and_unfrozen_selection_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_steps must be a positive integer"):
            self.quiet_main([*self.smoke, "--max-steps", "0"])
        self.assertFalse(self.run.exists())
        with self.assertRaisesRegex(ValueError, "selection.json required"):
            self.quiet_main(["evaluate", *self.common, "--checkpoint", str(self.run / "best.safetensors"),
                             "--output", str(self.root / "unselected.json")])

    def test_legacy_config_is_rejected_before_training(self):
        config = json.loads(self.config.read_text())
        config["head_lr"] = 0.001
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "legacy AgentJev configuration fields"):
            self.quiet_main(self.smoke)
        self.assertFalse(self.run.exists())


if __name__ == "__main__":
    unittest.main()
