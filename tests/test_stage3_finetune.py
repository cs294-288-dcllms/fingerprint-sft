import inspect
import os
import tempfile
import unittest
from pathlib import Path

from trl import SFTConfig

from stages.stage3_finetune import (
    _configure_runtime_caches,
    _mask_prompt,
    _record_sequence_length,
    _validate_training_responses,
    build_parser,
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
    def test_configures_writable_accelerator_caches_before_fla_import(self):
        variables = (
            "RUNTIME_CACHE_ROOT",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "CUDA_CACHE_PATH",
        )
        previous = {variable: os.environ.get(variable) for variable in variables}
        try:
            with tempfile.TemporaryDirectory() as temporary:
                for variable in variables:
                    os.environ.pop(variable, None)
                os.environ["RUNTIME_CACHE_ROOT"] = temporary

                configured = _configure_runtime_caches()

                self.assertEqual(
                    configured["TRITON_CACHE_DIR"],
                    Path(temporary) / "triton-sft",
                )
                self.assertEqual(
                    configured["TORCHINDUCTOR_CACHE_DIR"],
                    Path(temporary) / "torchinductor-sft",
                )
                self.assertEqual(
                    configured["CUDA_CACHE_PATH"],
                    Path(temporary) / "cuda",
                )
                self.assertTrue(all(path.is_dir() for path in configured.values()))
        finally:
            for variable, value in previous.items():
                if value is None:
                    os.environ.pop(variable, None)
                else:
                    os.environ[variable] = value

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
        row = _record_sequence_length({"input_ids": [1, 2, 3], "labels": [-100, 2, 3]})
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

    def test_max_steps_defaults_to_full_epoch_training(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--traces",
                "traces.jsonl",
                "--student-model",
                "student",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(args.max_steps, -1)

    def test_save_only_final_is_opt_in(self):
        parser = build_parser()
        common = [
            "--traces",
            "traces.jsonl",
            "--student-model",
            "student",
            "--output-dir",
            "output",
        ]
        self.assertFalse(parser.parse_args(common).save_only_final)
        self.assertTrue(parser.parse_args([*common, "--save-only-final"]).save_only_final)

    def test_save_only_final_disables_intermediate_checkpoints(self):
        source = inspect.getsource(run_stage3)
        self.assertIn(
            'save_strategy="no" if cfg.save_only_final else "steps"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
