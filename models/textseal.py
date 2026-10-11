"""TextSeal generation helpers backed by the official TextSeal PRF."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from textseal.watermarking.core import fast_prf_dual, prf_dual
from transformers import LogitsProcessor


@dataclass(frozen=True, slots=True)
class TextSealSpec:
    """Parameters for dual-key TextSeal generation."""

    secret_key: int
    secret_key_b: int
    ngram: int = 3
    mixing_alpha: float = 0.1
    temperature: float = 0.7
    top_p: float = 0.95
    initial_candidate_count: int = 4096
    seed: int = 42

    def __post_init__(self) -> None:
        if self.ngram <= 0:
            raise ValueError("ngram must be positive")
        if not 0.0 <= self.mixing_alpha <= 0.5:
            raise ValueError("mixing_alpha must be in [0, 0.5]")
        if self.temperature <= 0:
            raise ValueError("TextSeal requires temperature > 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.initial_candidate_count <= 0:
            raise ValueError("initial_candidate_count must be positive")


def _candidate_distribution(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    initial_candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an exact top-p candidate set without sorting the full vocabulary."""
    scaled_logits = logits.float() / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    vocabulary_size = probabilities.shape[-1]
    candidate_count = min(initial_candidate_count, vocabulary_size)

    while True:
        candidate_probs, candidate_ids = torch.topk(
            probabilities,
            k=candidate_count,
            dim=-1,
            largest=True,
            sorted=True,
        )
        if top_p >= 1.0 or bool((candidate_probs.sum(dim=-1) >= top_p).all()):
            break
        if candidate_count >= vocabulary_size:
            break
        candidate_count = min(vocabulary_size, candidate_count * 2)

    if top_p < 1.0:
        cumulative = torch.cumsum(candidate_probs, dim=-1)
        outside_nucleus = cumulative - candidate_probs > top_p
        candidate_probs = candidate_probs.masked_fill(outside_nucleus, 0.0)
    return candidate_probs, candidate_ids


def select_textseal_tokens(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    spec: TextSealSpec,
    *,
    use_key_a: torch.Tensor,
) -> torch.Tensor:
    """Select TextSeal tokens using dual-key Gumbel-max sampling."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocabulary]")
    if contexts.shape != (logits.shape[0], spec.ngram):
        raise ValueError(
            f"contexts must have shape {(logits.shape[0], spec.ngram)}, "
            f"received {tuple(contexts.shape)}"
        )
    if use_key_a.shape != (logits.shape[0],):
        raise ValueError(f"use_key_a must have shape {(logits.shape[0],)}")

    candidate_probs, candidate_ids = _candidate_distribution(
        logits,
        temperature=spec.temperature,
        top_p=spec.top_p,
        initial_candidate_count=spec.initial_candidate_count,
    )
    if logits.is_cuda:
        scores_a, scores_b = fast_prf_dual(
            contexts,
            candidate_ids,
            spec.secret_key,
            spec.secret_key_b,
        )
    else:
        scores_a, scores_b = prf_dual(
            contexts.unsqueeze(1),
            candidate_ids,
            spec.secret_key,
            spec.secret_key_b,
        )
    routed_scores = torch.where(use_key_a.unsqueeze(1), scores_a, scores_b)
    gumbel_scores = torch.log(routed_scores + 1e-30) / (candidate_probs + 1e-30)
    selected_offsets = torch.argmax(gumbel_scores, dim=-1, keepdim=True)
    return torch.gather(candidate_ids, dim=-1, index=selected_offsets).squeeze(-1)


def textseal_fused_scores(
    contexts: torch.Tensor,
    token_ids: torch.Tensor,
    spec: TextSealSpec,
) -> torch.Tensor:
    """Compute TextSeal's dual-key early-fusion score for token/context pairs."""
    if contexts.ndim != 2 or contexts.shape[-1] != spec.ngram:
        raise ValueError(f"contexts must have shape [N, {spec.ngram}]")
    if token_ids.ndim != 1 or token_ids.shape[0] != contexts.shape[0]:
        raise ValueError("token_ids must have shape [N]")
    listed_tokens = token_ids.unsqueeze(1)
    if contexts.is_cuda:
        scores_a, scores_b = fast_prf_dual(
            contexts,
            listed_tokens,
            spec.secret_key,
            spec.secret_key_b,
        )
    else:
        scores_a, scores_b = prf_dual(
            contexts.unsqueeze(1),
            listed_tokens,
            spec.secret_key,
            spec.secret_key_b,
        )
    score_a = -torch.log1p(-scores_a.clamp(max=1.0 - 1e-12))
    score_b = -torch.log1p(-scores_b.clamp(max=1.0 - 1e-12))
    return (spec.mixing_alpha * score_a + (1.0 - spec.mixing_alpha) * score_b).squeeze(1)


class TextSealLogitsProcessor(LogitsProcessor):
    """Force TextSeal-selected tokens inside cached batched ``generate()``."""

    def __init__(self, spec: TextSealSpec) -> None:
        self.spec = spec
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

    def _routing_generator(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.spec.seed)
            self._generator_device = device
        return self._generator

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        if input_ids.shape[-1] < self.spec.ngram:
            raise ValueError(
                f"TextSeal needs at least {self.spec.ngram} context tokens, "
                f"received {input_ids.shape[-1]}"
            )
        contexts = input_ids[:, -self.spec.ngram :]
        routing = (
            torch.rand(
                scores.shape[0],
                generator=self._routing_generator(scores.device),
                device=scores.device,
            )
            < self.spec.mixing_alpha
        )
        selected = select_textseal_tokens(
            scores,
            contexts,
            self.spec,
            use_key_a=routing,
        )
        forced = torch.full_like(scores, float("-inf"))
        return forced.scatter(1, selected.unsqueeze(1), 0.0)

    def reset(self) -> None:
        """Retain the RNG stream across batches; no per-batch state must be reset."""
