"""SemIf Direct choice scoring with the unchanged pretrained causal language model.

The Direct method is documented in SemIf-OpenJev at commit
23cf1f39fc9534fe81437200959b6dfc7106e45a. This module adapts that inference
method for this project's supervised training: it selects the native next-token
logits of the answer-option tokens and trains with restricted-choice CE.
There is no separately initialized classifier, reranker, or NLI model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SemIfDirectModel(nn.Module):
    """Select answer-option logits from a native Qwen3ForCausalLM.

    Prompts use left padding, so the final column is always the last real prompt
    token. Explicit position IDs count only real tokens, making a prompt's
    prediction independent of the amount of left padding in its batch.
    The wrapper registers only the supplied causal LM's original parameters.
    """

    def __init__(self, causal_lm: nn.Module):
        super().__init__()
        self.causal_lm = causal_lm

    @classmethod
    def from_pretrained(cls, path: str, *, revision: str | None = None,
                        local_files_only: bool = False,
                        attn_implementation: str | None = None) -> SemIfDirectModel:
        """Load the pretrained Qwen3 decoder, embeddings, and native LM head.

        No additional trainable module is initialized. Original tied embeddings
        remain tied when the checkpoint's configuration specifies that tie.
        Parameters are float32; mixed-precision autocast belongs to the caller.
        """
        from transformers import Qwen3ForCausalLM

        options = {"revision": revision, "local_files_only": local_files_only,
                   "torch_dtype": torch.float32, "output_loading_info": True,
                   "use_safetensors": True}
        if attn_implementation is not None:
            options["attn_implementation"] = attn_implementation
        causal_lm, loading_info = Qwen3ForCausalLM.from_pretrained(path, **options)
        failures = {key: loading_info.get(key) for key in
                    ("missing_keys", "mismatched_keys", "error_msgs") if loading_info.get(key)}
        if failures:
            raise ValueError(f"Incomplete native causal-LM checkpoint: {failures}")
        return cls(causal_lm)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]
        option_ids = batch["option_token_ids"]
        cand_mask = batch["cand_mask"]
        if input_ids.dtype != torch.long or input_ids.ndim != 2 or not all(input_ids.shape):
            raise ValueError("input_ids must be a nonempty [questions, tokens] long tensor")
        if attention_mask.shape != input_ids.shape or position_ids.shape != input_ids.shape:
            raise ValueError("attention_mask and position_ids must match input_ids")
        if position_ids.dtype != torch.long:
            raise ValueError("position_ids must be a long tensor")
        if any(value.device != input_ids.device for value in
               (attention_mask, position_ids, option_ids, cand_mask)):
            raise ValueError("All batch tensors must be on the same device")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("attention_mask must contain only zero and one")
        valid = attention_mask.bool()
        if not valid[:, -1].all() or (valid[:, :-1] & ~valid[:, 1:]).any():
            raise ValueError("Prompts must use contiguous left padding with a valid final token")
        expected_positions = valid.long().cumsum(-1) - 1
        if (position_ids < 0).any() or not torch.equal(position_ids[valid], expected_positions[valid]):
            raise ValueError("Real-token position_ids must count from zero, excluding left padding")
        vocab_size = self.causal_lm.config.vocab_size
        if (input_ids < 0).any() or (input_ids >= vocab_size).any():
            raise ValueError("input_ids contain a token outside the model vocabulary")
        if (option_ids.dtype != torch.long or option_ids.ndim != 2
                or option_ids.shape[0] != input_ids.shape[0] or option_ids.shape[1] == 0):
            raise ValueError("option_token_ids must be a [questions, options] long tensor")
        if cand_mask.dtype != torch.bool or cand_mask.shape != option_ids.shape:
            raise ValueError("cand_mask must be a boolean tensor matching option_token_ids")
        if not cand_mask.any(-1).all():
            raise ValueError("Every question must have at least one valid option")
        if (option_ids < 0).any() or (option_ids >= vocab_size).any():
            raise ValueError("option_token_ids contain a token outside the model vocabulary")
        for row, mask in zip(option_ids, cand_mask):
            if row[mask].unique().numel() != mask.sum().item():
                raise ValueError("Valid answer options must use distinct token IDs")
        output = self.causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        logits = output.logits[:, -1, :].gather(-1, option_ids)
        return {"logits": logits.masked_fill(~cand_mask, float("-inf")),
                "cand_mask": cand_mask}


def supervised_loss(logits: torch.Tensor, target: torch.Tensor,
                    cand_mask: torch.Tensor) -> torch.Tensor:
    """Mean cross entropy after normalizing over valid answer options only.

    Targets are one-hot labels or probability distributions in the same order
    as the option logits. Invalid options must have zero target mass. No Brier
    term or vocabulary-wide next-token loss is included.
    """
    if logits.ndim != 2 or target.shape != logits.shape or cand_mask.shape != logits.shape:
        raise ValueError("logits, target and cand_mask must share [questions, options] shape")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point")
    if cand_mask.dtype != torch.bool:
        raise ValueError("cand_mask must be boolean")
    if logits.shape[0] == 0 or not cand_mask.any(-1).all():
        raise ValueError("Every question must have at least one valid option")
    mask = cand_mask.to(device=logits.device)
    truth = target.to(device=logits.device, dtype=torch.float32)
    if not torch.isfinite(truth).all() or (truth < 0).any():
        raise ValueError("target must contain finite, nonnegative probabilities")
    if truth.masked_select(~mask).ne(0).any():
        raise ValueError("Invalid options must have zero target probability")
    if not torch.allclose(truth.sum(-1), torch.ones(logits.shape[0], device=logits.device),
                          atol=1e-5, rtol=1e-5):
        raise ValueError("Each target distribution must sum to one")
    if not torch.isfinite(logits.masked_select(mask)).all():
        raise ValueError("Valid option logits must be finite")
    masked_logits = logits.float().masked_fill(~mask, float("-inf"))
    log_probabilities = torch.log_softmax(masked_logits, -1).masked_fill(~mask, 0.0)
    return -(truth * log_probabilities).sum(-1).mean()
