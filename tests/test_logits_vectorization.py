from types import SimpleNamespace

import torch

from hashing import BigramHash, HashConfig
from models.logits import ADSLogitsProcessor, RadioactiveLogitsProcessor


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
