"""Numerical and interrupted-run checks for the actual LLaMA-Factory backend."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tests"))
DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "llamafactory", "tensorboard"))
if DEPENDENCIES:
    import torch
    from llamafactory.hparams import FinetuningArguments
    from safetensors.torch import load_file
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from test_jev_data import fixture_records, write_bundle
    from transformers import Qwen3Config, Qwen3ForCausalLM, TrainerCallback

    from customer_policy_verifier.training import train_jev as training
    from customer_policy_verifier.training.factory_trainer import (
        DirectCollator,
        DirectTrainer,
    )
    from customer_policy_verifier.training.factory_workflow import training_arguments
    from customer_policy_verifier.training.factory_workflow import training_plan
    from customer_policy_verifier.training.jev_data import (
        encode_record,
        load_dataset_bundle,
    )


@unittest.skipUnless(DEPENDENCIES, "full training dependencies required")
class FactoryTrainingTests(unittest.TestCase):
    def test_ratio_warmup_uses_real_optimizer_budget_and_cosine_scheduler(self):
        cfg = dict(self.cfg, max_steps=-1, num_train_epochs=1, warmup_steps=0,
                   warmup_ratio=0.1, gradient_accumulation=16, learning_rate=1e-5)
        row = {"input_ids": [1, 2], "option_token_ids": [32, 33], "target": [1.0, 0.0]}
        for count, steps, warmup in ((2442, 153, 16), (7227, 452, 46)):
            trainer = self.trainer(self.model(), cfg, self.root / f"plan-{count}", [row] * count)
            plan = training_plan(trainer)
            self.assertEqual(plan["optimizer_steps"], steps)
            self.assertEqual(plan["warmup_steps"], warmup)
            trainer.create_optimizer_and_scheduler(steps)
            schedule = trainer.lr_scheduler.lr_lambdas[0]
            self.assertEqual(schedule(0), 0.0)
            self.assertAlmostEqual(schedule(warmup), 1.0)
            self.assertEqual(schedule(steps), 0.0)

    def test_lowercase_checkpoint_reloads_and_evaluates_with_the_saved_transform(self):
        self.cfg.update(input_normalization="lowercase", num_train_epochs=1, max_steps=-1,
                        warmup_steps=0, warmup_ratio=0.1, eval_strategy="epoch")
        self.config.write_text(json.dumps(self.cfg))
        run = self.root / "lowercase"
        argv = ["train", "--config", str(self.config), "--data-dir", str(self.data),
                "--train-set", "mixed", "--run-dir", str(run)]
        args = training.parser().parse_args(argv)
        args.tiny_model = True
        with contextlib.redirect_stdout(io.StringIO()):
            selected = training.train(args, training.config_for(args), load_dataset_bundle(self.data))
        self.assertEqual(selected["completed_epochs"], 1.0)
        saved = json.loads((run / "run.json").read_text())
        self.assertEqual(saved["training_plan"]["warmup_steps"], 1)
        self.assertEqual(saved["input_protocol"]["input_normalization"]["name"], "lowercase")
        events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
        self.assertEqual([event["step"] for event in events if "eval_cross_entropy" in event], [2])
        scalars = EventAccumulator(str(run / "tensorboard")).Reload()
        self.assertEqual([event.step for event in scalars.Scalars("eval/cross_entropy")], [0, 2])
        output = self.root / "test-report.json"
        # Default CLI config remains the historical raw-input configuration.
        # Evaluation must obtain normalization from the checkpoint, not it.
        result = self.quiet(["evaluate", "--checkpoint", str(run / "best.safetensors"),
                             "--data-dir", str(self.data), "--split", "test",
                             "--device", "cpu", "--precision", "fp32", "--output", str(output)])
        self.assertEqual(result["input_protocol"]["input_normalization"]["name"], "lowercase")
        self.assertEqual(result["metrics"]["n"], 1)
        with patch.object(training, "encode_record", wraps=encode_record) as encoder:
            evaluation = training.parser().parse_args(["evaluate", "--checkpoint", str(run / "best.safetensors"),
                              "--data-dir", str(self.data), "--split", "test", "--output", str(self.root / "again.json")])
            evaluation.device, evaluation.precision = "cpu", "fp32"
            training.evaluate_checkpoint(evaluation, load_dataset_bundle(self.data))
        self.assertEqual(encoder.call_args.args[3], "lowercase")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cfg = json.loads(training.DEFAULT_CONFIG.read_text())
        self.cfg.update(device="cpu", precision="fp32", max_steps=2, eval_every=1,
                        max_tokens=256, warmup_steps=0, gradient_accumulation=2,
                        microbatch_questions=1, eval_batch_size=2, learning_rate=0.001)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(self.cfg))
        self.data = self.root / "data"
        self.data.mkdir()
        write_bundle(self.data, fixture_records())

    def command(self, run):
        return ["smoke", "--tiny-model", "--samples", "3", "--train-set", "mixed",
                "--config", str(self.config), "--data-dir", str(self.data), "--run-dir", str(run)]

    def quiet(self, argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return training.main(argv)

    def model(self):
        torch.manual_seed(42)
        cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, tie_word_embeddings=True, max_position_embeddings=256)
        cfg._attn_implementation = "sdpa"
        cfg.use_cache = False
        return Qwen3ForCausalLM(cfg)

    def trainer(self, model, cfg, directory, rows, callbacks=()):
        return DirectTrainer(model=model, args=training_arguments(cfg, directory, smoke=True),
                             finetuning_args=FinetuningArguments(stage="sft", finetuning_type="full"),
                             processor=None, tokenizer=None, data_collator=DirectCollator(0),
                             train_dataset=rows, eval_dataset=rows, callbacks=list(callbacks))

    def test_loss_uses_only_two_option_logits_without_a_vocabulary_target(self):
        model = self.model().eval()
        rows = [{"input_ids": [1, 2, 3], "option_token_ids": [32, 33], "target": [0.0, 1.0]}]
        trainer = self.trainer(model, self.cfg, self.root / "loss", rows)
        batch = DirectCollator(0)(rows)
        loss, output = trainer.compute_loss(model, batch, return_outputs=True)
        native_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              position_ids=batch["position_ids"], logits_to_keep=1).logits[:, -1, :]
        expected = torch.nn.functional.cross_entropy(native_logits[:, [32, 33]], torch.tensor([1]))
        vocabulary_loss = torch.nn.functional.cross_entropy(native_logits, torch.tensor([33]))
        torch.testing.assert_close(loss, expected)
        self.assertGreater(float(vocabulary_loss.detach() - loss.detach()), 2.0)
        self.assertEqual(output["logits"].shape, (1, 2))

    def test_accumulated_gradients_match_one_full_batch(self):
        rows = [{"input_ids": [1, 2, 3], "option_token_ids": [32, 33], "target": [1.0, 0.0]},
                {"input_ids": [1, 4, 5, 6], "option_token_ids": [32, 33], "target": [0.0, 1.0]}]
        gradients = []

        class Capture(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                gradients.append({name: p.grad.detach().clone() for name, p in model.named_parameters()})

        for microbatch, accumulation in ((2, 1), (1, 2)):
            cfg = dict(self.cfg, max_steps=1, grad_clip=1e6, microbatch_questions=microbatch,
                       gradient_accumulation=accumulation)
            trainer = self.trainer(self.model(), cfg, self.root / f"batch-{microbatch}", rows, [Capture()])
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train()
        self.assertEqual(len(gradients), 2)
        self.assertEqual(gradients[0].keys(), gradients[1].keys())
        for name in gradients[0]:
            with self.subTest(parameter=name):
                self.assertTrue(torch.isfinite(gradients[1][name]).all())
                torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-4, atol=1e-6)

    def interrupt(self, run):
        original = DirectTrainer.train

        class StopAfterOne(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_training_stop = True
                return control

        def stopped(trainer, *args, **kwargs):
            trainer.add_callback(StopAfterOne())
            return original(trainer, *args, **kwargs)

        with patch.object(DirectTrainer, "train", stopped), self.assertRaisesRegex(RuntimeError, "stopped before max_steps"):
            self.quiet(self.command(run))
        self.assertFalse((run / "selection.json").exists())
        return run / "checkpoints/checkpoint-1"

    def test_resume_restores_the_same_updates_as_an_uninterrupted_run(self):
        full, interrupted = self.root / "full", self.root / "resumed"
        self.quiet(self.command(full))
        checkpoint = self.interrupt(interrupted)
        restored_steps = []
        original = DirectTrainer.train

        class Capture(TrainerCallback):
            def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
                restored_steps.extend(float(v["step"]) for v in optimizer.state.values())

        def capture(trainer, *args, **kwargs):
            trainer.add_callback(Capture())
            return original(trainer, *args, **kwargs)

        with patch.object(DirectTrainer, "train", capture):
            result = self.quiet([*self.command(interrupted), "--resume-from-checkpoint", str(checkpoint)])
        self.assertEqual(result["completed_steps"], 2)
        self.assertTrue(restored_steps and all(step == 1 for step in restored_steps))
        left = load_file(full / "checkpoints/checkpoint-2/model.safetensors")
        right = load_file(interrupted / "checkpoints/checkpoint-2/model.safetensors")
        for name in left:
            with self.subTest(parameter=name):
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        self.assertEqual(json.loads((full / "selection.json").read_text())["best_step"], result["best_step"])
        self.assertTrue(json.loads((interrupted / "reload_check.json").read_text())["passed"])

    def test_resume_rejects_changed_sample_subset_before_loading_weights(self):
        run = self.root / "interrupted"
        checkpoint = self.interrupt(run)
        with (
            patch.object(training, "build_model", side_effect=AssertionError("unexpected model loading")),
            self.assertRaisesRegex(ValueError, "differs from the original run"),
        ):
            self.quiet([*self.command(run), "--samples", "2", "--resume-from-checkpoint", str(checkpoint)])

    def test_resume_recovers_selection_after_a_later_incomplete_save(self):
        run = self.root / "selection-interrupted"
        checkpoint = self.interrupt(run)
        expected = training.sha256(checkpoint / "cpv_best.safetensors")
        replacement = run / "new-best.tmp"
        replacement.write_bytes(b"a later incomplete save replaced the mutable best file")
        replacement.replace(run / "best.safetensors")
        self.assertNotEqual(training.sha256(run / "best.safetensors"), expected)
        self.assertEqual(training.sha256(checkpoint / "cpv_best.safetensors"), expected)
        result = self.quiet([*self.command(run), "--resume-from-checkpoint", str(checkpoint)])
        self.assertEqual(result["completed_steps"], 2)
        self.assertTrue(json.loads((run / "reload_check.json").read_text())["passed"])

    def test_export_is_the_selected_native_qwen_model_with_tied_weights(self):
        run = self.root / "export-run"
        self.quiet(self.command(run))
        args = training.parser().parse_args(self.command(run))
        selected, _, _ = training.reload_checkpoint(run / "best.safetensors", args)
        exported = Qwen3ForCausalLM.from_pretrained(run / "export", local_files_only=True,
                                                  use_safetensors=True)
        self.assertIs(exported.lm_head.weight, exported.model.embed_tokens.weight)
        self.assertEqual(selected.causal_lm.state_dict().keys(), exported.state_dict().keys())
        for name, weight in exported.state_dict().items():
            torch.testing.assert_close(weight, selected.causal_lm.state_dict()[name], rtol=0, atol=0)

    def test_completed_updates_can_finish_export_after_an_interruption(self):
        run = self.root / "export-interrupted"
        original = Qwen3ForCausalLM.save_pretrained

        def fail_export(model, directory, **kwargs):
            if Path(directory).name == "export":
                raise InterruptedError("export interrupted")
            return original(model, directory, **kwargs)

        with patch.object(Qwen3ForCausalLM, "save_pretrained", fail_export), self.assertRaisesRegex(InterruptedError, "export interrupted"):
            self.quiet(self.command(run))
        self.assertFalse((run / "selection.json").exists())
        with patch.object(DirectTrainer, "train", side_effect=AssertionError("must not train extra updates")):
            result = self.quiet([*self.command(run), "--resume-from-checkpoint",
                                 str(run / "checkpoints/checkpoint-2")])
        self.assertEqual(result["completed_steps"], 2)
        self.assertTrue((run / "export/model.safetensors").is_file())
    def epoch_run(self, run, checkpoint=None):
        # Exercise the ordinary epoch workflow with offline random weights.
        # The production CLI intentionally only permits --tiny-model for smoke.
        argv = ["train", "--train-set", "mixed", "--config", str(self.config),
                "--data-dir", str(self.data), "--run-dir", str(run), "--num-train-epochs", "3"]
        if checkpoint:
            argv.extend(["--resume-from-checkpoint", str(checkpoint)])
        args = training.parser().parse_args(argv)
        args.tiny_model = True
        with contextlib.redirect_stdout(io.StringIO()):
            return training.train(args, training.config_for(args), load_dataset_bundle(self.data))

    def test_three_epochs_consume_all_samples_and_save_final_validation_and_tensorboard(self):
        run = self.root / "epochs"
        self.cfg.update(eval_every=100, logging_every=1)
        self.config.write_text(json.dumps(self.cfg))
        seen = []
        original = DirectTrainer.training_step

        def capture(trainer, model, inputs, *args, **kwargs):
            seen.extend(tuple(row.tolist()) for row in inputs["input_ids"])
            return original(trainer, model, inputs, *args, **kwargs)

        with patch.object(DirectTrainer, "training_step", capture):
            selected = self.epoch_run(run)
        records = load_dataset_bundle(self.data)["train"]
        expected = Counter(tuple(encode_record(row, training.TinyTokenizer(), 256)["input_ids"])
                           for row in records)
        self.assertEqual(Counter(seen), Counter({key: count * 3 for key, count in expected.items()}))
        self.assertEqual(selected["completed_steps"], 6, "three rows form two accumulation windows per epoch")
        self.assertEqual(selected["completed_epochs"], 3.0)
        saved = json.loads((run / "run.json").read_text())
        self.assertEqual(saved["framework_arguments"]["max_steps"], -1)
        self.assertEqual(saved["training_plan"]["mode"], "epochs")
        self.assertEqual(saved["training_plan"]["optimizer_steps"], 6)
        self.assertTrue((run / "checkpoints/checkpoint-6/cpv_resume.json").is_file())
        events = EventAccumulator(str(run / "tensorboard")).Reload()
        self.assertTrue({"train/loss", "train/learning_rate", "train/grad_norm",
                         "eval/accuracy", "eval/macro_f1", "eval/cross_entropy"}
                        <= set(events.Tags()["scalars"]))
        self.assertEqual([item.step for item in events.Scalars("eval/cross_entropy")], [0, 6])
        json_events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
        final = next(item for item in json_events if item["step"] == 6 and "eval_cross_entropy" in item)
        self.assertAlmostEqual(events.Scalars("eval/cross_entropy")[-1].value,
                               final["eval_cross_entropy"], places=6)
        self.assertIn("eval_confusion_matrix", final)
        self.assertNotIn("eval/confusion_matrix", events.Tags()["scalars"])

    def test_epoch_resume_matches_full_weights_and_preserves_tensorboard_steps(self):
        full, interrupted = self.root / "epoch-full", self.root / "epoch-resumed"
        self.cfg.update(logging_every=1)
        self.config.write_text(json.dumps(self.cfg))
        self.epoch_run(full)
        original = DirectTrainer.train

        class StopAtThree(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 3:
                    control.should_training_stop = True
                return control

        def stop(trainer, *args, **kwargs):
            trainer.add_callback(StopAtThree())
            return original(trainer, *args, **kwargs)

        with patch.object(DirectTrainer, "train", stop), self.assertRaisesRegex(RuntimeError, "stopped before max_steps"):
            self.epoch_run(interrupted)
        result = self.epoch_run(interrupted, interrupted / "checkpoints/checkpoint-3")
        self.assertEqual(result["completed_steps"], 6)
        self.assertEqual(result["completed_epochs"], 3.0)
        left = load_file(full / "checkpoints/checkpoint-6/model.safetensors")
        right = load_file(interrupted / "checkpoints/checkpoint-6/model.safetensors")
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        events = EventAccumulator(str(interrupted / "tensorboard")).Reload()
        self.assertEqual([item.step for item in events.Scalars("train/loss")], [1, 2, 3, 4, 5, 6])
        self.assertGreaterEqual(len(list((interrupted / "tensorboard").glob("events.out.tfevents.*"))), 2)

    def test_completed_epoch_checkpoint_can_finish_export_without_extra_updates(self):
        run = self.root / "epoch-export-interrupted"
        original = Qwen3ForCausalLM.save_pretrained

        def fail_export(model, directory, **kwargs):
            if Path(directory).name == "export":
                raise InterruptedError("export interrupted")
            return original(model, directory, **kwargs)

        with patch.object(Qwen3ForCausalLM, "save_pretrained", fail_export), self.assertRaisesRegex(InterruptedError, "export interrupted"):
            self.epoch_run(run)
        with patch.object(DirectTrainer, "train", side_effect=AssertionError("must not train more epochs")):
            result = self.epoch_run(run, run / "checkpoints/checkpoint-6")
        self.assertEqual(result["completed_steps"], 6)
        self.assertEqual(result["completed_epochs"], 3.0)
        self.assertTrue(json.loads((run / "reload_check.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
