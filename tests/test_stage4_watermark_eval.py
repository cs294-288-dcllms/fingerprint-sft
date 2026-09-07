import json
import tempfile
import unittest
from pathlib import Path

from stages.stage4_watermark_eval import (
    _append_stage4_checkpoint,
    _effective_batch_size,
    _load_stage4_checkpoint,
)


class Stage4BatchGuardTest(unittest.TestCase):
    def test_caps_qwen35_science_batch(self) -> None:
        self.assertEqual(
            _effective_batch_size(12, "science", "Qwen/Qwen3.5-4B"),
            4,
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


if __name__ == "__main__":
    unittest.main()
