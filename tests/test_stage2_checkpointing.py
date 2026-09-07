import json
import tempfile
import unittest
from pathlib import Path

from stages.stage2_teacher_eval import (
    _append_stage2_checkpoint,
    _load_stage2_checkpoint,
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


if __name__ == "__main__":
    unittest.main()
