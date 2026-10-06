"""Offline numerical Direct-method tests with tiny Qwen3 models only."""

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEPENDENCIES = (importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("transformers") is not None)
if DEPENDENCIES:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from customer_policy_verifier.training.jev_model import SemIfDirectModel, supervised_loss


@unittest.skipUnless(DEPENDENCIES, "torch and transformers are required for tiny-model numerical tests")
class SemIfDirectModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.config = Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=32, attention_dropout=0.0,
            tie_word_embeddings=True, pad_token_id=0,
        )
        self.native = Qwen3ForCausalLM(self.config)
        self.model = SemIfDirectModel(self.native)
        self.batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [0, 0, 9, 10]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]]),
            "position_ids": torch.tensor([[0, 1, 2, 3], [0, 0, 0, 1]]),
            "option_token_ids": torch.tensor([[20, 21], [20, 21]]),
            "cand_mask": torch.tensor([[True, True], [True, True]]),
        }

    def test_direct_scores_match_native_selected_vocabulary_logits(self):
        self.model.eval()
        with torch.no_grad():
            full = self.native(**{key: self.batch[key] for key in
                                  ("input_ids", "attention_mask", "position_ids")},
                               use_cache=False, return_dict=True, logits_to_keep=0).logits
            with patch.object(self.native, "forward", wraps=self.native.forward) as forward:
                result = self.model(self.batch)
        expected = full[:, -1, :].gather(-1, self.batch["option_token_ids"])
        torch.testing.assert_close(result["logits"], expected)
        self.assertEqual(tuple(result["logits"].shape), (2, 2))
        self.assertTrue(torch.equal(result["cand_mask"], self.batch["cand_mask"]))
        self.assertEqual(forward.call_args.kwargs["logits_to_keep"], 1)
        self.assertFalse(forward.call_args.kwargs["use_cache"])

    def test_wrapper_adds_no_trainable_parameters_and_preserves_native_head(self):
        self.assertIs(self.model.causal_lm, self.native)
        self.assertEqual({id(p) for p in self.model.parameters()}, {id(p) for p in self.native.parameters()})
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),
                         sum(p.numel() for p in self.native.parameters()))
        self.assertIs(self.native.lm_head.weight, self.native.model.embed_tokens.weight)
        self.assertTrue(all(name.startswith("causal_lm.") for name, _ in self.model.named_parameters()))

    def test_four_choice_scores_use_the_same_native_head_without_binary_restriction(self):
        batch = {**self.batch, "option_token_ids": torch.tensor([[20, 21, 22, 23], [20, 21, 22, 23]]),
                 "cand_mask": torch.ones((2, 4), dtype=torch.bool)}
        self.model.eval()
        with torch.no_grad():
            native = self.native(**{key: batch[key] for key in
                                   ("input_ids", "attention_mask", "position_ids")},
                                 use_cache=False, return_dict=True).logits[:, -1, :]
            scored = self.model(batch)["logits"]
        torch.testing.assert_close(scored, native.gather(-1, batch["option_token_ids"]))
        self.assertEqual(tuple(scored.shape), (2, 4))

    def test_full_parameter_gradients_and_checkpoint_reload(self):
        result = self.model(self.batch)
        truth = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        loss = supervised_loss(result["logits"], truth, result["cand_mask"])
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertTrue(parameter.requires_grad, name)
            self.assertEqual(parameter.dtype, torch.float32, name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for module in (self.native.model.embed_tokens, self.native.model.layers[0].self_attn,
                       self.native.model.layers[0].mlp, self.native.lm_head):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
        restored = SemIfDirectModel(Qwen3ForCausalLM(self.config))
        restored.load_state_dict(self.model.state_dict(), strict=True)
        self.model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.batch)["logits"], restored(self.batch)["logits"])

    def test_left_padding_does_not_change_prompt_scores(self):
        self.model.eval()
        alone = {key: value[1:2] for key, value in self.batch.items()}
        for key in ("input_ids", "attention_mask", "position_ids"):
            alone[key] = alone[key][:, -2:]
        altered = dict(self.batch)
        altered["input_ids"] = self.batch["input_ids"].clone()
        altered["input_ids"][1, :2] = 31
        with torch.no_grad():
            original = self.model(self.batch)["logits"]
            torch.testing.assert_close(self.model(alone)["logits"], original[1:2], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(self.model(altered)["logits"], original, atol=1e-6, rtol=1e-5)

    def test_slot_permutation_and_mask_follow_native_logits(self):
        permuted = dict(self.batch)
        permuted["option_token_ids"] = self.batch["option_token_ids"].flip(-1)
        masked = dict(self.batch)
        masked["cand_mask"] = torch.tensor([[True, False], [True, True]])
        self.model.eval()
        with torch.no_grad():
            original = self.model(self.batch)["logits"]
            torch.testing.assert_close(self.model(permuted)["logits"], original.flip(-1))
            result = self.model(masked)["logits"]
        self.assertTrue(torch.isneginf(result[0, 1]))
        torch.testing.assert_close(result[masked["cand_mask"]], original[masked["cand_mask"]])

    def test_pretrained_helper_loads_native_complete_model_in_float32(self):
        with patch("transformers.Qwen3ForCausalLM.from_pretrained", return_value=(self.native, {})) as load:
            initialized = SemIfDirectModel.from_pretrained(
                "unused-offline-path", revision="fixed-revision", local_files_only=True,
                attn_implementation="eager",
            )
        load.assert_called_once_with("unused-offline-path", revision="fixed-revision",
                                     local_files_only=True, torch_dtype=torch.float32, output_loading_info=True,
                                     use_safetensors=True,
                                     attn_implementation="eager")
        self.assertIs(initialized.causal_lm, self.native)

    def test_pretrained_helper_rejects_missing_mismatched_or_failed_native_weights(self):
        for key in ("missing_keys", "mismatched_keys", "error_msgs"):
            with self.subTest(key=key):
                with patch("transformers.Qwen3ForCausalLM.from_pretrained",
                           return_value=(self.native, {key: ["lm_head.weight"]})):
                    with self.assertRaisesRegex(ValueError, "Incomplete native causal-LM checkpoint"):
                        SemIfDirectModel.from_pretrained("unused-offline-path", local_files_only=True)

    def test_real_local_pretrained_reload_preserves_weights_and_embedding_tie(self):
        self.native.eval()
        with tempfile.TemporaryDirectory() as tmp:
            self.native.save_pretrained(tmp)
            loaded = SemIfDirectModel.from_pretrained(tmp, local_files_only=True,
                                                      attn_implementation="eager")
        self.assertIs(loaded.causal_lm.lm_head.weight, loaded.causal_lm.model.embed_tokens.weight)
        self.assertTrue(all(p.dtype == torch.float32 and p.requires_grad for p in loaded.parameters()))
        for name, value in self.native.state_dict().items():
            torch.testing.assert_close(loaded.causal_lm.state_dict()[name], value)
        loaded.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(self.batch)["logits"], self.model(self.batch)["logits"],
                                       atol=1e-6, rtol=1e-5)

    def test_bad_prompt_padding_positions_and_option_slots_are_rejected(self):
        cases = [
            ("attention_mask", torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])),
            ("attention_mask", torch.tensor([[1, 0, 1, 1], [0, 0, 1, 1]])),
            ("attention_mask", torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])),
            ("attention_mask", torch.tensor([[1, 2, 1, 1], [0, 0, 1, 1]])),
            ("attention_mask", torch.ones((2, 3), dtype=torch.long)),
            ("position_ids", torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])),
            ("position_ids", torch.zeros((2, 4), dtype=torch.float32)),
            ("input_ids", torch.zeros((2, 4), dtype=torch.float32)),
            ("input_ids", torch.tensor([[1, 2, 3, 32], [0, 0, 9, 10]])),
            ("option_token_ids", torch.tensor([[20, 20], [20, 21]])),
            ("option_token_ids", torch.tensor([[20, 32], [20, 21]])),
            ("option_token_ids", torch.tensor([[20, -1], [20, 21]])),
            ("option_token_ids", torch.tensor([[20.0, 21.0], [20.0, 21.0]])),
            ("option_token_ids", torch.tensor([[20, 21]])),
            ("cand_mask", torch.tensor([[0, 0], [1, 1]], dtype=torch.bool)),
            ("cand_mask", torch.ones((2, 2), dtype=torch.long)),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                batch = {**self.batch, key: value}
                with self.assertRaises(ValueError):
                    self.model(batch)

    def test_masked_cross_entropy_matches_analytic_value_and_finite_gradients(self):
        logits = torch.tensor([[math.log(3), 0.0, float("nan")]], requires_grad=True)
        truth = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        loss = supervised_loss(logits, truth, mask)
        self.assertAlmostEqual(loss.item(), -math.log(0.75), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 2].item(), 0.0)

    def test_restricted_cross_entropy_matches_one_hot_and_soft_targets(self):
        logits = torch.tensor([[1.0, -1.0], [0.4, 0.9]], requires_grad=True)
        mask = torch.ones_like(logits, dtype=torch.bool)
        truth = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        torch.testing.assert_close(supervised_loss(logits, truth, mask),
                                   torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])))
        soft = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
        expected = -(soft * torch.log_softmax(logits, -1)).sum(-1).mean()
        torch.testing.assert_close(supervised_loss(logits, soft, mask), expected)

    def test_invalid_loss_inputs_fail_before_softmax(self):
        logits = torch.tensor([[0.5, 0.2, float("inf")]])
        truth = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        cases = [
            (truth, torch.zeros_like(mask)),
            (torch.tensor([[0.5, 0.0, 0.0]]), mask),
            (torch.tensor([[0.9, -0.1, 0.2]]), mask),
            (torch.tensor([[0.5, 0.0, 0.5]]), mask),
            (torch.tensor([[float("nan"), 0.0, 0.0]]), mask),
            (truth, mask.to(torch.int64)),
        ]
        for target, candidate_mask in cases:
            with self.subTest(target=target, mask=candidate_mask):
                with self.assertRaises(ValueError):
                    supervised_loss(logits, target, candidate_mask)
        with self.assertRaises(ValueError):
            supervised_loss(logits, truth, torch.ones_like(mask))
        with self.assertRaises(ValueError):
            supervised_loss(logits, truth[:, :2], mask)


if __name__ == "__main__":
    unittest.main()
