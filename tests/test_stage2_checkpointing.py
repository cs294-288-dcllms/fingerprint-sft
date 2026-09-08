import json
import tempfile
import unittest
from pathlib import Path

from stages.stage2_teacher_eval import (
    _append_stage2_checkpoint,
    _load_stage2_checkpoint,
    _validate_stage2_completion,
)


class Stage2CheckpointingTest(unittest.TestCase):
    def test_checkpoint_resumes_and_repairs_partial_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rank_000.jsonl"
            _append_stage2_checkpoint(path, [0, 1], 1, 2, 2)
            _append_stage2_checkpoint(path, [2], 1, 0, 1)
            with path.open("ab") as handle:
                handle.write(b'{"positions": [3], "raw_correct":')

            positions, raw, forced, total = _load_stage2_checkpoint(path)

            self.assertEqual(positions, {0, 1, 2})
            self.assertEqual((raw, forced, total), (2, 2, 3))
            for line in path.read_text(encoding="utf-8").splitlines():
                json.loads(line)

    def test_completion_requires_exact_positions_and_count(self) -> None:
        _validate_stage2_completion({0, 1}, expected_positions=2, total=2)
        with self.assertRaisesRegex(RuntimeError, r"missing_positions=\[1\]"):
            _validate_stage2_completion({0}, expected_positions=2, total=1)
        with self.assertRaisesRegex(RuntimeError, r"total=3"):
            _validate_stage2_completion({0, 1}, expected_positions=2, total=3)


if __name__ == "__main__":
    unittest.main()
