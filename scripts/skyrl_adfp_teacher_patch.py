#!/usr/bin/env python3
"""Runtime patch that applies the existing ADS transform to SkyRL's ref logits."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


class BigramMask:
    """Exact vectorized hash used by the local ADFP implementation."""

    def __init__(
        self,
        *,
        seed: int,
        gamma: float,
        vocab_size: int,
        excluded_token_ids: list[int],
    ) -> None:
        self.seed = int(seed) % (2**63)
        self.gamma = float(gamma)
        self.vocab_size = int(vocab_size)
        self.excluded = torch.tensor(
            sorted({int(token_id) for token_id in excluded_token_ids}),
            dtype=torch.long,
        )
        self._cache: dict[str, tuple[torch.Tensor, ...]] = {}

    def _tensors(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        key = str(device)
        if key not in self._cache:
            mul1 = torch.tensor(6364136223846793005, device=device, dtype=torch.int64)
            mul2 = torch.tensor(1442695040888963407, device=device, dtype=torch.int64)
            mul3 = torch.tensor(22695477, device=device, dtype=torch.int64)
            token_base = (
                torch.arange(self.vocab_size, device=device, dtype=torch.int64) * mul1
            )
            self._cache[key] = (
                token_base,
                mul2,
                mul3,
                torch.tensor((1 << 63) - 1, device=device, dtype=torch.int64),
                torch.tensor(self.seed, device=device, dtype=torch.int64),
                self.excluded.to(device),
            )
        return self._cache[key]

    def mask_batch(
        self,
        bigrams: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        device = bigrams.device
        token_base, mul2, mul3, mask63, seed, excluded = self._tensors(device)
        bigrams = bigrams.to(dtype=torch.long)
        values = token_base.unsqueeze(0) ^ (bigrams[:, :1] * mul2)
        values = values ^ (bigrams[:, 1:] * mul3) ^ seed
        values = (values ^ (values >> 30)) * mul2
        values = (values ^ (values >> 27)) * mul3
        values = (values ^ (values >> 31)) & mask63
        if self.gamma == 0.5:
            mask = values < ((1 << 62) - 256)
        else:
            mask = values.to(torch.float64) / float(2**63) < self.gamma
        if excluded.numel() > 0:
            mask.index_fill_(1, excluded, False)
        return mask.to(dtype=dtype)


def _bigrams_for_positions(
    sequences: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    current = sequences[:, start:end]
    if start == 0:
        previous = torch.cat(
            [
                sequences.new_full((sequences.shape[0], 1), -1),
                sequences[:, : max(end - 1, 0)],
            ],
            dim=1,
        )
    else:
        previous = sequences[:, start - 1 : end - 1]
    return torch.stack((previous, current), dim=-1)


def install_adfp_teacher_patch() -> None:
    """Patch HFModelWrapper once in every driver/worker interpreter."""

    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

    if getattr(HFModelWrapper, "_adfp_teacher_patch_installed", False):
        return

    original_init = HFModelWrapper.__init__
    original_forward = HFModelWrapper.forward

    def patched_init(self, pretrain_or_model, *args, **kwargs):
        original_init(self, pretrain_or_model, *args, **kwargs)
        enabled = os.environ.get("SKYRL_ADFP_TEACHER_ENABLED", "0") == "1"
        teacher_path = os.environ.get("SKYRL_ADFP_TEACHER_PATH", "")
        if not enabled or not isinstance(pretrain_or_model, str):
            self._adfp_teacher_enabled = False
            return
        if Path(pretrain_or_model).resolve() != Path(teacher_path).resolve():
            self._adfp_teacher_enabled = False
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        proxy_path = os.environ["SKYRL_ADFP_PROXY_MODEL"]
        hash_path = Path(os.environ["SKYRL_ADFP_HASH_CONFIG"])
        hash_config = json.loads(hash_path.read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(
            pretrain_or_model,
            trust_remote_code=True,
            local_files_only=True,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        proxy = AutoModelForCausalLM.from_pretrained(
            proxy_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        proxy.config.use_cache = False
        proxy.requires_grad_(False)
        proxy.eval()
        proxy.to(device)

        self.adfp_proxy_model = proxy
        self._adfp_teacher_enabled = True
        self._adfp_lambda = float(os.environ.get("SKYRL_ADFP_LAMBDA", "16"))
        self._adfp_position_chunk = int(
            os.environ.get("SKYRL_ADFP_POSITION_CHUNK", "8")
        )
        self._adfp_eos_token_id = tokenizer.eos_token_id
        self._adfp_sequences: torch.Tensor | None = None
        self._adfp_attention_mask: torch.Tensor | None = None
        self._adfp_mask = BigramMask(
            seed=int(hash_config["seed"]),
            gamma=float(hash_config["gamma"]),
            vocab_size=int(self.model.config.vocab_size),
            excluded_token_ids=list(tokenizer.all_special_ids),
        )

        def apply_ads(_module, _args, _kwargs, output):
            sequences = self._adfp_sequences
            if sequences is None:
                return output
            teacher_logits = output["logits"]
            attention_mask = self._adfp_attention_mask
            if attention_mask is None:
                attention_mask = torch.ones_like(sequences)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            with torch.inference_mode():
                proxy_output = self.adfp_proxy_model(
                    input_ids=sequences,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                proxy_logits = proxy_output["logits"]
                if proxy_logits.shape != teacher_logits.shape:
                    raise RuntimeError(
                        "ADFP proxy/teacher logits are not aligned: "
                        f"proxy={tuple(proxy_logits.shape)} "
                        f"teacher={tuple(teacher_logits.shape)}"
                    )
                sequence_length = teacher_logits.shape[1]
                for start in range(
                    0,
                    sequence_length,
                    self._adfp_position_chunk,
                ):
                    end = min(start + self._adfp_position_chunk, sequence_length)
                    proxy_probs = F.softmax(
                        proxy_logits[:, start:end, :],
                        dim=-1,
                    )
                    bigrams = _bigrams_for_positions(sequences, start, end)
                    flat_bigrams = bigrams.reshape(-1, 2)
                    mask = self._adfp_mask.mask_batch(
                        flat_bigrams,
                        dtype=proxy_probs.dtype,
                    ).reshape_as(proxy_probs)
                    mask_probability = (proxy_probs * mask).sum(
                        dim=-1,
                        keepdim=True,
                    )
                    ad_term = proxy_probs * (mask - mask_probability)
                    if self._adfp_eos_token_id is not None:
                        active = teacher_logits[:, start:end, :].argmax(
                            dim=-1
                        ).ne(self._adfp_eos_token_id)
                        ad_term.mul_(active.unsqueeze(-1))
                    teacher_logits[:, start:end, :].add_(
                        ad_term.to(teacher_logits.dtype),
                        alpha=self._adfp_lambda,
                    )
                del proxy_output, proxy_logits
            output["logits"] = teacher_logits
            return output

        self.model.register_forward_hook(apply_ads, with_kwargs=True)

    def patched_forward(self, sequences, *args, **kwargs):
        if getattr(self, "_adfp_teacher_enabled", False):
            if self.remove_microbatch_padding or self.sequence_parallel_size != 1:
                raise RuntimeError(
                    "ADFP teacher scoring requires unpacked sequences and sequence_parallel_size=1"
                )
            self._adfp_sequences = sequences
            self._adfp_attention_mask = kwargs.get("attention_mask")
        try:
            return original_forward(self, sequences, *args, **kwargs)
        finally:
            if getattr(self, "_adfp_teacher_enabled", False):
                self._adfp_sequences = None
                self._adfp_attention_mask = None

    HFModelWrapper.__init__ = patched_init
    HFModelWrapper.forward = patched_forward
    HFModelWrapper._adfp_teacher_patch_installed = True


__all__ = ["BigramMask", "install_adfp_teacher_patch"]
