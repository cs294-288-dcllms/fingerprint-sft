"""Radioactive and ADS logits processors (aligned with comparison implementation)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from hashing import BigramHash


def _extract_bigram(row: torch.LongTensor) -> Tuple[int, int]:
    """Return the last two token ids (or -1 placeholders).

    Args:
        row: 1D tensor of token ids.

    Returns:
        Tuple of (prev_id, next_id), using -1 for missing positions.
    """
    length = row.shape[0]
    if length >= 2:
        return int(row[-2].item()), int(row[-1].item())
    if length == 1:
        return -1, int(row[-1].item())
    return -1, -1


def _extract_bigram_batch(input_ids: torch.LongTensor) -> torch.LongTensor:
    """Return the last two token ids for every row without GPU synchronization."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    batch_size, sequence_length = input_ids.shape
    if sequence_length >= 2:
        return input_ids[:, -2:]
    bigrams = input_ids.new_full((batch_size, 2), -1)
    if sequence_length == 1:
        bigrams[:, 1] = input_ids[:, 0]
    return bigrams


class RadioactiveLogitsProcessor(LogitsProcessor):
    """Increase logits for every token that satisfies the hash predicate."""

    def __init__(self, hash_fn: BigramHash, delta: float):
        """Create a radioactive logits processor.

        Args:
            hash_fn: BigramHash used to compute green lists.
            delta: Additive logit boost for green-list tokens.

        Returns:
            None.
        """
        super().__init__()
        self.hash_fn = hash_fn
        self.delta = float(delta)
        self.eos_token_id: int | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply radioactive perturbation to a batch of logits.

        Args:
            input_ids: Tensor of token ids for the current batch.
            scores: Logits tensor to be modified in-place.

        Returns:
            Updated logits tensor.
        """
        if self.delta == 0:
            return scores
        bigrams = _extract_bigram_batch(input_ids)
        mask = self.hash_fn.mask_batch(
            bigrams,
            device=scores.device,
            dtype=scores.dtype,
        )
        if self.eos_token_id is not None and self.eos_token_id < scores.shape[-1]:
            active = scores.argmax(dim=-1).ne(self.eos_token_id)
            mask = mask * active.unsqueeze(-1)
        return scores + mask * self.delta


class CachedProxyModel:
    """KV-cache helper for proxy forward passes (assumes shared tokenizer)."""

    def __init__(self, model, *, pad_token_id: int):
        """Wrap a model to reuse KV-cache across incremental calls.

        Args:
            model: Causal LM model to forward.
            pad_token_id: Token id used for padding attention masks.

        Returns:
            None.
        """
        self.model = model
        self.pad_token_id = int(pad_token_id)
        self.past_key_values = None
        self.cached_length = 0

    def reset(self) -> None:
        """Clear cached KV state.

        Returns:
            None.
        """
        self.past_key_values = None
        self.cached_length = 0

    @torch.inference_mode()
    def __call__(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """Forward pass with caching to avoid recomputing prefix states.

        Args:
            input_ids: Token ids for the current sequence batch.

        Returns:
            Logits tensor for the provided input ids.
        """
        attention_mask = input_ids.ne(self.pad_token_id).long()
        needs_full_pass = (
            self.past_key_values is None or input_ids.shape[1] <= self.cached_length
        )

        if needs_full_pass:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            self.cached_length = input_ids.shape[1]
        else:
            new_token = input_ids[:, -1:]
            outputs = self.model(
                input_ids=new_token,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=self.past_key_values,
                return_dict=True,
            )
            self.cached_length += 1

        self.past_key_values = outputs.past_key_values
        return outputs.logits


class ADSLogitsProcessor(LogitsProcessor):
    """Apply ADS perturbations using proxy distribution (assumes tokenizer alignment)."""

    def __init__(
        self,
        hash_fn: BigramHash,
        lam: float,
        proxy_model,
        *,
        pad_token_id: int,
        eos_token_id: int | None = None,
    ):
        """Create an ADS logits processor.

        Args:
            hash_fn: BigramHash used to compute green lists.
            lam: ADS strength parameter.
            proxy_model: Proxy model or CachedProxyModel instance.
            pad_token_id: Padding token id for attention masks.
            eos_token_id: Optional EOS id used to skip perturbations.

        Returns:
            None.
        """
        super().__init__()
        self.hash_fn = hash_fn
        self.lam = float(lam)
        self.proxy = (
            proxy_model
            if isinstance(proxy_model, CachedProxyModel)
            else CachedProxyModel(proxy_model, pad_token_id=pad_token_id)
        )
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id) if eos_token_id is not None else None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply ADS perturbation using the proxy distribution.

        Args:
            input_ids: Tensor of token ids for the current batch.
            scores: Logits tensor to be modified in-place.

        Returns:
            Updated logits tensor.
        """
        if self.lam == 0:
            return scores

        proxy_logits = self.proxy(input_ids)[:, -1, :]
        proxy_probs = F.softmax(proxy_logits, dim=-1)

        bigrams = _extract_bigram_batch(input_ids)
        mask = self.hash_fn.mask_batch(
            bigrams,
            device=scores.device,
            dtype=proxy_probs.dtype,
        )
        mask_prob = (proxy_probs * mask).sum(dim=-1, keepdim=True)
        ad_term = proxy_probs * (mask - mask_prob)
        if self.eos_token_id is not None and self.eos_token_id < scores.shape[-1]:
            active = scores.argmax(dim=-1).ne(self.eos_token_id)
            ad_term = ad_term * active.unsqueeze(-1)
        return scores + self.lam * ad_term.to(scores.dtype)


__all__ = [
    "RadioactiveLogitsProcessor",
    "ADSLogitsProcessor",
    "CachedProxyModel",
]
