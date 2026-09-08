from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import GenerationConfig, ModelSpec
from hashing import HashConfig
from data.science import ScienceProvider
from stages.stage1_generate import (
    _append_checkpoint_rows,
    _build_metadata_payload,
    _load_checkpoint_rows,
    _validate_merged_rows,
)
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

    def test_merge_requires_exact_unique_index_coverage(self) -> None:
        rows = [
            {"index": 1, "response": "one"},
            {"index": 0, "response": "zero"},
        ]
        self.assertEqual(
            [row["index"] for row in _validate_merged_rows(rows, 2)],
            [0, 1],
        )

        with self.assertRaisesRegex(RuntimeError, r"missing_indices=\[1\]"):
            _validate_merged_rows([rows[1]], 2)
        with self.assertRaisesRegex(RuntimeError, r"duplicate_indices=\[0\]"):
            _validate_merged_rows([rows[1], rows[1]], 2)

    def test_metadata_records_generation_seed_and_sampling_settings(self) -> None:
        cfg = GenerationConfig(
            dataset="science",
            split="train",
            max_examples=9_000,
            teacher=ModelSpec(name="teacher"),
            proxy=ModelSpec(name="proxy"),
            method="ads",
            lam=16,
            seed=43,
            max_new_tokens=3_840,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.0,
        )

        metadata = _build_metadata_payload(
            cfg,
            9_000,
            hash_cfg=HashConfig(seed=123, gamma=0.5),
            trace_sha256="a" * 64,
        )

        self.assertEqual(metadata["seed"], 43)
        self.assertEqual(metadata["temperature"], 0.7)
        self.assertEqual(metadata["top_p"], 0.95)
        self.assertEqual(metadata["repetition_penalty"], 1.0)
        self.assertEqual(metadata["teacher_model"], "teacher")
        self.assertEqual(metadata["teacher_dtype"], "bfloat16")
        self.assertEqual(metadata["proxy_model"], "proxy")
        self.assertEqual(metadata["proxy_dtype"], "bfloat16")
        self.assertEqual(metadata["hash_seed"], 123)
        self.assertEqual(metadata["hash_gamma"], 0.5)
        self.assertEqual(metadata["trace_sha256"], "a" * 64)


class ScienceTeacherEvalTest(unittest.TestCase):
    def test_matches_boxed_or_plain_final_choices(self) -> None:
        self.assertTrue(_is_science_correct("Final answer: C", "\\boxed{C}"))
        self.assertTrue(_is_science_correct("reasoning... \\boxed{B}", "\\boxed{B}"))
        self.assertFalse(_is_science_correct("\\boxed{A}", "\\boxed{D}"))


if __name__ == "__main__":
    unittest.main()
