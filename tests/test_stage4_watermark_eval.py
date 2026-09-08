import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from hashing import BigramHash, HashConfig
from stages.stage4_watermark_eval import (
    _aggregate_stage4_shards,
    _aligned_offsets,
    _append_stage4_checkpoint,
    _build_metric_provenance,
    _build_position_batches,
    _effective_batch_size,
    _filter_first_occurrences,
    _has_identity_token_mapping,
    _load_stage4_checkpoint,
    _validate_stage4_completion,
)


class Stage4BatchGuardTest(unittest.TestCase):
    def test_metric_provenance_hashes_exact_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = root / "traces.jsonl"
            hash_config = root / "hash.json"
            adapter = root / "adapter"
            traces.write_text('{"response":"x"}\n', encoding="utf-8")
            hash_config.write_text('{"seed":1,"gamma":0.5}\n', encoding="utf-8")
            adapter.mkdir()
            cfg = SimpleNamespace(
                dataset="science",
                traces_jsonl=traces,
                hash_config=hash_config,
                lora_dir=adapter,
                seed=43,
            )

            provenance = _build_metric_provenance(cfg, 1)

            self.assertEqual(provenance["dataset"], "science")
            self.assertEqual(provenance["trace_file"], str(traces.resolve()))
            self.assertEqual(provenance["trace_examples"], 1)
            self.assertEqual(len(provenance["trace_sha256"]), 64)
            self.assertEqual(
                provenance["hash_config_file"],
                str(hash_config.resolve()),
            )
            self.assertEqual(len(provenance["hash_config_sha256"]), 64)
            self.assertEqual(
                provenance["student_lora_dir"],
                str(adapter.resolve()),
            )
            self.assertEqual(provenance["seed"], 43)

    def test_left_padding_preserves_actual_token_positions(self) -> None:
        offsets = [(0, 0), (0, 0), (0, 1), (1, 3), (3, 6)]
        attention_mask = [0, 0, 1, 1, 1]

        self.assertEqual(
            _aligned_offsets(offsets, attention_mask),
            {1: 2, 3: 3, 6: 4},
        )

    def test_alignment_rejects_mismatched_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            _aligned_offsets([(0, 1)], [1, 1])

    def test_caps_qwen35_science_batch(self) -> None:
        self.assertEqual(
            _effective_batch_size(12, "science", "Qwen/Qwen3.5-4B"),
            12,
        )

    def test_qwen35_science_batches_preserve_order_and_token_budget(self) -> None:
        positions = list(range(10))
        token_lengths = [4096] * 5 + [1000] * 5

        batches = _build_position_batches(
            positions,
            token_lengths,
            12,
            "science",
            "Qwen/Qwen3.5-4B",
        )

        self.assertEqual(
            batches,
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]],
        )
        self.assertEqual(
            [position for batch in batches for position in batch],
            positions,
        )

    def test_other_workloads_use_fixed_batches(self) -> None:
        self.assertEqual(
            _build_position_batches(
                list(range(7)),
                [1000] * 7,
                3,
                "gsm8k",
                "Qwen/Qwen3.5-4B",
            ),
            [[0, 1, 2], [3, 4, 5], [6]],
        )

    def test_preserves_other_workloads(self) -> None:
        self.assertEqual(
            _effective_batch_size(12, "gsm8k", "Qwen/Qwen3.5-4B"),
            12,
        )
        self.assertEqual(
            _effective_batch_size(12, "science", "Qwen/Qwen2.5-3B"),
            12,
        )

    def test_enforces_positive_batch(self) -> None:
        self.assertEqual(
            _effective_batch_size(0, "science", "Qwen/Qwen3.5-4B"),
            1,
        )

    def test_detects_exact_identity_token_mapping(self) -> None:
        mapping = torch.arange(5)
        shared = torch.ones(5, dtype=torch.bool)

        self.assertTrue(_has_identity_token_mapping(mapping, shared, 5, 5))
        self.assertFalse(
            _has_identity_token_mapping(
                torch.tensor([0, 1, 3, 2, 4]),
                shared,
                5,
                5,
            )
        )
        self.assertFalse(
            _has_identity_token_mapping(
                mapping,
                torch.tensor([True, True, False, True, True]),
                5,
                5,
            )
        )

    def test_sparse_hash_membership_matches_dense_gather(self) -> None:
        bigrams = torch.tensor(
            [[1, 2], [7, 11], [13, 17], [19, 23], [29, 31]],
            dtype=torch.long,
        )
        token_ids = torch.tensor([0, 3, 17, 128, 256], dtype=torch.long)
        for gamma in (0.5, 0.37):
            hash_fn = BigramHash(
                HashConfig(seed=294288, gamma=gamma),
                vocab_size=257,
                excluded_token_ids=[0, 256],
            )
            dense = hash_fn.mask_batch(bigrams)
            expected = dense.gather(1, token_ids.unsqueeze(1)).squeeze(1)

            actual = hash_fn.membership_batch(bigrams, token_ids)

            self.assertTrue(torch.equal(actual, expected))

    def test_filters_rank_local_bigrams_by_first_occurrence(self) -> None:
        seen = {(1, 2)}

        bigrams, positions, samples = _filter_first_occurrences(
            [(1, 2), (2, 3), (2, 3), (3, 4)],
            [10, 11, 12, 13],
            [0, 0, 1, 1],
            seen,
        )

        self.assertEqual(bigrams, [(2, 3), (3, 4)])
        self.assertEqual(positions, [11, 13])
        self.assertEqual(samples, [0, 1])
        self.assertEqual(seen, {(1, 2), (2, 3), (3, 4)})

    def test_first_occurrence_filter_requires_aligned_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "differ in length"):
            _filter_first_occurrences(
                [(1, 2)],
                [],
                [0],
                set(),
            )


    def test_checkpoint_resumes_and_repairs_partial_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rank_000.jsonl"
            _append_stage4_checkpoint(path, [0, 1], [((1, 2), 0.6)])
            _append_stage4_checkpoint(path, [2], [((2, 3), 0.7)])
            with path.open("ab") as handle:
                handle.write(b'{"positions": [3], "entries":')

            positions, entries = _load_stage4_checkpoint(path)

            self.assertEqual(positions, {0, 1, 2})
            self.assertEqual(entries, [((1, 2), 0.6), ((2, 3), 0.7)])
            for line in path.read_text(encoding="utf-8").splitlines():
                json.loads(line)

    def test_completion_requires_exact_local_positions(self) -> None:
        _validate_stage4_completion({0, 1}, expected_positions=2)
        with self.assertRaisesRegex(RuntimeError, r"missing_positions=\[1\]"):
            _validate_stage4_completion({0}, expected_positions=2)
        with self.assertRaisesRegex(RuntimeError, r"unexpected_positions=\[2\]"):
            _validate_stage4_completion({0, 1, 2}, expected_positions=2)

    def test_shard_aggregation_preserves_first_occurrence_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "rank_000.jsonl"
            second = Path(temporary) / "rank_001.jsonl"
            _append_stage4_checkpoint(
                first,
                [0],
                [((1, 2), 0.25), ((2, 3), 0.5)],
            )
            _append_stage4_checkpoint(
                second,
                [0],
                [((1, 2), 0.9), ((3, 4), 1.0)],
            )

            count, mean = _aggregate_stage4_shards([first, second])

            self.assertEqual(count, 3)
            self.assertAlmostEqual(mean, (0.25 + 0.5 + 1.0) / 3)


if __name__ == "__main__":
    unittest.main()
