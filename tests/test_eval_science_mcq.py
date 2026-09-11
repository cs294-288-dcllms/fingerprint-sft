from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_science_mcq.py"
SPEC = importlib.util.spec_from_file_location("eval_science_mcq", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class EvalScienceMcqTest(unittest.TestCase):
    def test_boxed_choice(self) -> None:
        self.assertEqual(MODULE.extract_choice("Therefore \\boxed{c}."), "C")

    def test_answer_fallback(self) -> None:
        self.assertEqual(MODULE.extract_choice("The final answer is B."), "B")

    def test_scoring(self) -> None:
        self.assertTrue(MODULE.is_correct("Answer: D", "work \\boxed{D}"))
        self.assertFalse(MODULE.is_correct("Answer: A", "work \\boxed{D}"))

    def test_merge_requires_exact_unique_index_coverage(self) -> None:
        rows = [{"index": 1}, {"index": 0}]
        self.assertEqual(
            [row["index"] for row in MODULE._validate_complete_predictions(rows, 2)],
            [0, 1],
        )
        with self.assertRaisesRegex(RuntimeError, r"missing_indices=\[1\]"):
            MODULE._validate_complete_predictions([rows[1]], 2)
        with self.assertRaisesRegex(RuntimeError, r"duplicate_indices=\[0\]"):
            MODULE._validate_complete_predictions([rows[1], rows[1]], 2)

    def test_runtime_caches_are_created_before_model_import(self) -> None:
        cache_variables = (
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "CUDA_CACHE_PATH",
        )
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(
                os.environ,
                {"RUNTIME_CACHE_ROOT": temporary},
                clear=False,
            ):
                for variable in cache_variables:
                    os.environ.pop(variable, None)
                configured = MODULE.configure_runtime_caches()
                self.assertEqual(set(configured), set(cache_variables))
                self.assertTrue(all(path.is_dir() for path in configured.values()))


if __name__ == "__main__":
    unittest.main()
