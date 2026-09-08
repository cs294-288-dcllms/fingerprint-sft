import inspect
import tempfile
import unittest

from trl import SFTConfig

from stages.stage3_finetune import (
    _mask_prompt,
    _record_sequence_length,
    _validate_training_responses,
    run_stage3,
)


class FakeTokenizer:
    eos_token_id = 9

    def __call__(self, text, add_special_tokens=False):
        values = {
            "prompt": [1, 2],
            "response": [3, 4],
            "ended": [3, 9],
        }
        return {"input_ids": list(values[text])}


class Stage3MaskingTest(unittest.TestCase):
    def test_requires_nonempty_response_for_every_trace(self):
        _validate_training_responses([{"response": "reasoning"}])
        with self.assertRaisesRegex(RuntimeError, "first_indices=\\[1, 2\\]"):
            _validate_training_responses(
                [{"response": "ok"}, {"response": ""}, {"prompt": "missing"}]
            )

    def test_appends_eos_to_response_labels(self):
        row = _mask_prompt(FakeTokenizer(), "prompt", "response", 16)
        self.assertEqual(row["input_ids"], [1, 2, 3, 4, 9])
        self.assertEqual(row["labels"], [-100, -100, 3, 4, 9])

    def test_does_not_duplicate_existing_eos(self):
        row = _mask_prompt(FakeTokenizer(), "prompt", "ended", 16)
        self.assertEqual(row["input_ids"], [1, 2, 3, 9])
        self.assertEqual(row["labels"], [-100, -100, 3, 9])

    def test_records_explicit_token_length_for_grouped_sampler(self):
        row = _record_sequence_length(
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
        )
        self.assertEqual(row["length"], 3)

    def test_groups_similar_sequence_lengths_for_training_throughput(self):
        source = inspect.getsource(run_stage3)
        self.assertIn('train_sampling_strategy="group_by_length"', source)
        self.assertIn('length_column_name="length"', source)
        with tempfile.TemporaryDirectory() as tmp:
            config = SFTConfig(
                output_dir=tmp,
                train_sampling_strategy="group_by_length",
                length_column_name="length",
            )
        self.assertEqual(config.train_sampling_strategy, "group_by_length")
        self.assertEqual(config.length_column_name, "length")


if __name__ == "__main__":
    unittest.main()
