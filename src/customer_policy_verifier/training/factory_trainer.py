"""LLaMA-Factory extension for Direct decisions rather than text generation.

The native causal LM remains the Trainer model, so framework checkpoints and
exports retain the standard Qwen3ForCausalLM layout and tied embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable

import torch
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from torch.utils.data import Sampler
from transformers.integrations import TensorBoardCallback

from customer_policy_verifier.training.jev_data import binary_metrics
from customer_policy_verifier.training.jev_model import (
    SemIfDirectModel,
    supervised_loss,
)


@dataclass
class DirectCollator:
    pad_token_id: int

    def __call__(self, rows):
        from customer_policy_verifier.training.train_jev import collate

        # Accelerate moves the batch to the selected device. String metadata is
        # deliberately omitted, and no answer token is appended to the prompt.
        return collate(rows, self.pad_token_id, "cpu")


class ScalarTensorBoardCallback(TensorBoardCallback):
    """Keep structured metrics in JSON while sending scalar curves to TensorBoard."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        scalars = {key: value for key, value in (logs or {}).items() if isinstance(value, Real)}
        return super().on_log(args, state, control, logs=scalars, **kwargs)


class EpochSeededSampler(Sampler):
    """Keep the saved epoch's shuffle when Accelerate wraps a skipped loader."""

    def __init__(self, dataset, seed: int, epoch: Callable[[], int]):
        self.dataset, self.seed, self.epoch = dataset, seed, epoch

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # Accelerate 1.11 rebuilds a skipped DataLoaderShard with iteration=0.
        # Reading Trainer's restored epoch avoids resetting a later shuffle to
        # epoch zero, without replacing the framework's training/resume loop.
        generator = torch.Generator().manual_seed(self.seed + self.epoch())
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())


class DirectTrainer(CustomSeq2SeqTrainer):
    """Override the two places where generic SFT assumes vocabulary labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "tensorboard" in self.args.report_to:
            self.pop_callback(TensorBoardCallback)
            self.add_callback(ScalarTensorBoardCallback())
        # Qwen accepts **kwargs, but our loss is a per-question mean and does
        # not consume HF's token-count normalization. Trainer must divide it
        # by the actual number of accumulated microbatches exactly once.
        self.model_accepts_loss_kwargs = False

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return EpochSeededSampler(dataset, seed, lambda: int(self.state.epoch or 0))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = SemIfDirectModel(model)(inputs)
        loss = supervised_loss(outputs["logits"], inputs["target"], inputs["cand_mask"])
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        # Only two logits per question reach the metric accumulator. No
        # generation, decoding, or [batch, sequence, vocabulary] retention.
        return loss.detach(), outputs["logits"].detach().float(), inputs["target"].detach()


def decision_metrics(prediction, threshold=0.5):
    logits = torch.as_tensor(prediction.predictions).float()
    targets = torch.as_tensor(prediction.label_ids).argmax(-1).tolist()
    return binary_metrics(targets, logits.softmax(-1).tolist(), threshold)
