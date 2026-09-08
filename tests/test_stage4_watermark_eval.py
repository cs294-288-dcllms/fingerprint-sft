import json
import tempfile
import unittest
from pathlib import Path

from stages.stage4_watermark_eval import (
    _aggregate_stage4_shards,
    _aligned_offsets,
    _append_stage4_checkpoint,
    _build_position_batches,
    _effective_batch_size,
    _load_stage4_checkpoint,
    _validate_stage4_completion,
)


class Stage4BatchGuardTest(unittest.TestCase):
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
