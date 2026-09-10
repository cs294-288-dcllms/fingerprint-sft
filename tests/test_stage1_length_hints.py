from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stages.stage1_generate import (
    _order_examples_by_length_hints,
    _resolve_length_hints,
)


def _write_traces(path: Path, responses: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for response in responses:
            json.dump({"response": response}, handle)
            handle.write("\n")


class LengthHintTests(unittest.TestCase):
    def test_auto_hints_prefer_lambda16_for_stronger_ads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "training_traces" / "ads-lambda32" / "traces.jsonl"
            hint = root / "training_traces" / "ads-lambda16" / "traces.jsonl"
            _write_traces(hint, ["a", "bbbb", "cc"])
            lengths, source = _resolve_length_hints(
                "auto", output_jsonl=output, expected_rows=3
            )
            self.assertEqual(lengths, [1, 4, 2])
            self.assertEqual(source, hint)

    def test_auto_hints_prefer_control_for_radioactive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "training_traces" / "radioactive-delta2" / "traces.jsonl"
            hint = root / "training_traces" / "control" / "traces.jsonl"
            _write_traces(hint, ["abc", "d"])
            lengths, source = _resolve_length_hints(
                "auto", output_jsonl=output, expected_rows=2
            )
            self.assertEqual(lengths, [3, 1])
            self.assertEqual(source, hint)

    def test_auto_hints_skip_incomplete_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "alternative_traces" / "ads-lambda32" / "traces.jsonl"
            incomplete = root / "alternative_traces" / "ads-lambda16" / "traces.jsonl"
            fallback = root / "alternative_traces" / "ads-lambda8" / "traces.jsonl"
            _write_traces(incomplete, ["only-one"])
            _write_traces(fallback, ["a", "bb"])
            lengths, source = _resolve_length_hints(
                "auto", output_jsonl=output, expected_rows=2
            )
            self.assertEqual(lengths, [1, 2])
            self.assertEqual(source, fallback)

    def test_explicit_invalid_hints_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "training_traces" / "ads-lambda32" / "traces.jsonl"
            hint = root / "bad.jsonl"
            _write_traces(hint, ["one"])
            with self.assertRaisesRegex(ValueError, "expected 2"):
                _resolve_length_hints(hint, output_jsonl=output, expected_rows=2)

    def test_length_order_retains_global_indices(self) -> None:
        examples = [(0, "zero"), (2, "two"), (4, "four")]
        hints = [30, 0, 10, 0, 20]
        ordered = _order_examples_by_length_hints(examples, hints)
        self.assertEqual(ordered, [(2, "two"), (4, "four"), (0, "zero")])


if __name__ == "__main__":
    unittest.main()
