from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.science import ScienceProvider
from stages.stage1_generate import _append_checkpoint_rows, _load_checkpoint_rows
from stages.stage2_teacher_eval import _is_science_correct


class ScienceProviderTest(unittest.TestCase):
    def test_loads_prompt_and_gold_choice_from_local_jsonl(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "Which option is correct? A: one B: two C: three D: four"},
                {"role": "assistant", "content": "<think>reason</think>\n\\boxed{C}"},
            ],
            "_meta": {"gold_choice": "C"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "science.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with patch.dict(os.environ, {"SCIENCE_TRAIN_PATH": str(path)}):
                examples = list(ScienceProvider().load("train", limit=1))

        self.assertEqual(examples[0].prompt, row["messages"][0]["content"])
        self.assertEqual(examples[0].solution, "\\boxed{C}")


class Stage1CheckpointTest(unittest.TestCase):
    def test_resumes_and_repairs_a_truncated_tail(self) -> None:
        first = {"index": 0, "response": "one", "solution": "A"}
        second = {"index": 8, "response": "two", "solution": "B"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank_000.jsonl"
            _append_checkpoint_rows(path, [first])
            with path.open("a", encoding="utf-8") as handle:
                handle.write('[]\n{"index":')

            self.assertEqual(_load_checkpoint_rows(path), [first])
            _append_checkpoint_rows(path, [second])
            self.assertEqual(_load_checkpoint_rows(path), [first, second])


class ScienceTeacherEvalTest(unittest.TestCase):
    def test_matches_boxed_or_plain_final_choices(self) -> None:
        self.assertTrue(_is_science_correct("Final answer: C", "\\boxed{C}"))
        self.assertTrue(_is_science_correct("reasoning... \\boxed{B}", "\\boxed{B}"))
        self.assertFalse(_is_science_correct("\\boxed{A}", "\\boxed{D}"))


if __name__ == "__main__":
    unittest.main()
