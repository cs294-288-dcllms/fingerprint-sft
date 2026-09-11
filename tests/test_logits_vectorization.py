from types import SimpleNamespace

import torch

from hashing import BigramHash, HashConfig
from models.logits import ADSLogitsProcessor, RadioactiveLogitsProcessor
from scripts.skyrl_adfp_teacher_patch import BigramMask, _bigrams_for_positions


class _DummyProxy(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        use_cache: bool = True,
        past_key_values=None,
        return_dict: bool = True,
    ):
        del attention_mask, use_cache, past_key_values, return_dict
        vocab = torch.arange(self.vocab_size, device=input_ids.device, dtype=torch.float32)
        logits = vocab.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        logits = logits + input_ids.unsqueeze(-1).to(torch.float32) / 1000
        return SimpleNamespace(logits=logits, past_key_values=("cached",))


def _hash(vocab_size: int) -> BigramHash:
    return BigramHash(
        HashConfig(seed=294288, gamma=0.5),
        vocab_size=vocab_size,
        excluded_token_ids=[0, 1],
    )


def test_tensor_mask_batch_matches_individual_masks() -> None:
    hash_fn = _hash(31)
    bigrams = torch.tensor([[2, 3], [5, 8], [-1, 13]], dtype=torch.long)

    batched = hash_fn.mask_batch(bigrams)
    individual = torch.stack(
        [hash_fn.mask((int(row[0]), int(row[1]))) for row in bigrams]
    )

    assert torch.equal(batched, individual)


def test_gamma_half_integer_threshold_matches_float64_boundary() -> None:
    center = 1 << 62
    values = center + torch.arange(-2048, 2049, dtype=torch.int64)

    float64_reference = values.to(torch.float64).div(float(2**63)).lt(0.5)
    integer_fast_path = values.lt(center - 256)

    assert torch.equal(integer_fast_path, float64_reference)


def test_online_opd_hash_and_sequence_bigrams_match_sft() -> None:
    bigrams = torch.tensor(
        [[-1, 0], [1, 2], [7, 11], [13, 17], [19, 23]],
        dtype=torch.long,
    )
    for gamma in (0.5, 0.37):
        reference = BigramHash(
            HashConfig(seed=294288, gamma=gamma),
            vocab_size=257,
            excluded_token_ids=[0, 256],
        )
        online = BigramMask(
            seed=294288,
            gamma=gamma,
            vocab_size=257,
            excluded_token_ids=[0, 256],
        )
        assert torch.equal(
            online.mask_batch(bigrams, dtype=torch.bool),
            reference.mask_batch(bigrams),
        )

    sequences = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    assert _bigrams_for_positions(sequences, 0, 4).tolist() == [
        [[-1, 10], [10, 11], [11, 12], [12, 13]],
        [[-1, 20], [20, 21], [21, 22], [22, 23]],
    ]
    assert _bigrams_for_positions(sequences, 2, 4).tolist() == [
        [[11, 12], [12, 13]],
        [[21, 22], [22, 23]],
    ]


def test_radioactive_vectorization_matches_rowwise_equation() -> None:
    hash_fn = _hash(23)
    input_ids = torch.tensor([[2, 4, 6], [3, 5, 7], [8, 9, 10]])
    scores = torch.linspace(-1, 1, steps=69, dtype=torch.float32).reshape(3, 23)
    processor = RadioactiveLogitsProcessor(hash_fn, delta=2.0)
    processor.eos_token_id = 22

    expected = scores.clone()
    for index, row in enumerate(input_ids):
        if expected[index].argmax().item() == processor.eos_token_id:
            continue
        expected[index] += hash_fn.mask(
            (int(row[-2]), int(row[-1])),
            dtype=expected.dtype,
        ) * processor.delta

    actual = processor(input_ids, scores.clone())

    assert torch.equal(actual, expected)


def test_ads_vectorization_matches_rowwise_equation() -> None:
    vocab_size = 29
    hash_fn = _hash(vocab_size)
    input_ids = torch.tensor([[2, 4, 6], [3, 5, 7], [8, 9, 10]])
    scores = torch.linspace(1, -1, steps=87, dtype=torch.float32).reshape(3, vocab_size)
    processor = ADSLogitsProcessor(
        hash_fn,
        lam=16,
        proxy_model=_DummyProxy(vocab_size),
        pad_token_id=0,
        eos_token_id=28,
    )

    proxy_logits = _DummyProxy(vocab_size)(input_ids).logits[:, -1, :]
    proxy_probs = torch.softmax(proxy_logits, dim=-1)
    expected = scores.clone()
    for index, row in enumerate(input_ids):
        if expected[index].argmax().item() == processor.eos_token_id:
            continue
        mask = hash_fn.mask(
            (int(row[-2]), int(row[-1])),
            dtype=proxy_probs.dtype,
        )
        mask_prob = torch.dot(proxy_probs[index], mask)
        expected[index] += processor.lam * proxy_probs[index] * (mask - mask_prob)

    actual = processor(input_ids, scores.clone())

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
