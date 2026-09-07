import tempfile
import unittest
from pathlib import Path

from stages.stage3_finetune import (
    SFT_SAVE_STEPS,
    SFT_SAVE_TOTAL_LIMIT,
    _latest_checkpoint,
)


class Stage3ResumeTest(unittest.TestCase):
    def test_checkpoint_policy_is_bounded(self) -> None:
        self.assertEqual(SFT_SAVE_STEPS, 50)
        self.assertEqual(SFT_SAVE_TOTAL_LIMIT, 2)

    def test_latest_checkpoint_selects_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "checkpoint-50").mkdir()
            latest = output_dir / "checkpoint-100"
            latest.mkdir()

            self.assertEqual(_latest_checkpoint(output_dir), str(latest))

    def test_missing_output_has_no_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(_latest_checkpoint(Path(temporary) / "missing"))


if __name__ == "__main__":
    unittest.main()
