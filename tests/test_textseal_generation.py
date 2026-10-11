from __future__ import annotations

import pytest
import torch

from models.textseal import TextSealSpec, _candidate_distribution, select_textseal_tokens


def _reference_select(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    spec: TextSealSpec,
    use_key_a: torch.Tensor,
) -> torch.Tensor:
    from textseal.watermarking.core import prf_dual

    probs = torch.softmax(logits.float() / spec.temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(probs_sort, dim=-1)
    probs_sort[cumulative - probs_sort > spec.top_p] = 0.0
    scores_a, scores_b = prf_dual(
        contexts.unsqueeze(1),
        probs_idx,
        spec.secret_key,
        spec.secret_key_b,
    )
    routed = torch.where(use_key_a.unsqueeze(1), scores_a, scores_b)
    offsets = torch.argmax(torch.log(routed + 1e-30) / (probs_sort + 1e-30), dim=-1)
    return probs_idx.gather(1, offsets.unsqueeze(1)).squeeze(1)


def test_adaptive_candidates_cover_exact_top_p_nucleus() -> None:
    torch.manual_seed(7)
    logits = torch.randn(3, 257)
    candidate_probs, _ = _candidate_distribution(
        logits,
        temperature=0.7,
        top_p=0.95,
        initial_candidate_count=8,
    )
    assert bool((candidate_probs.sum(dim=-1) >= 0.95).all())


def test_textseal_selection_matches_full_sort_reference() -> None:
    torch.manual_seed(11)
    logits = torch.randn(4, 509)
    contexts = torch.randint(0, 509, (4, 3))
    use_key_a = torch.tensor([True, False, False, True])
    spec = TextSealSpec(
        secret_key=294288,
        secret_key_b=306633,
        ngram=3,
        mixing_alpha=0.1,
        temperature=0.7,
        top_p=0.95,
        initial_candidate_count=16,
    )
    actual = select_textseal_tokens(logits, contexts, spec, use_key_a=use_key_a)
    expected = _reference_select(logits, contexts, spec, use_key_a)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("mixing_alpha", [0.0, 0.1, 0.5])
def test_textseal_accepts_paper_sweep_range(mixing_alpha: float) -> None:
    assert (
        TextSealSpec(
            secret_key=1,
            secret_key_b=2,
            mixing_alpha=mixing_alpha,
        ).mixing_alpha
        == mixing_alpha
    )


@pytest.mark.parametrize("mixing_alpha", [-0.01, 0.5001, 1.0])
def test_textseal_rejects_alpha_outside_paper_sweep_range(
    mixing_alpha: float,
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 0\.5\]"):
        TextSealSpec(
            secret_key=1,
            secret_key_b=2,
            mixing_alpha=mixing_alpha,
        )
