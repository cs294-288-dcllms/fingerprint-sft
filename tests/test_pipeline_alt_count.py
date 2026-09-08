import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PipelineAltCountTest(unittest.TestCase):
    def test_pipeline_uses_independent_alt_count(self) -> None:
        pipeline = (ROOT / "pipeline.sh").read_text(encoding="utf-8")

        self.assertIn(
            'ALT_NUM_EXAMPLES="${ALT_NUM_EXAMPLES:-${NUM_EXAMPLES}}"',
            pipeline,
        )
        self.assertIn('--max-examples "$ALT_NUM_EXAMPLES"', pipeline)
        self.assertGreaterEqual(pipeline.count("_n${ALT_NUM_EXAMPLES}"), 4)

    def test_science_config_uses_one_thousand_alt_examples(self) -> None:
        config = (ROOT / "configs" / "science-qwen35.env").read_text(
            encoding="utf-8"
        )

        self.assertIn('NUM_EXAMPLES="${NUM_EXAMPLES:-9000}"', config)
        self.assertIn('ALT_NUM_EXAMPLES="${ALT_NUM_EXAMPLES:-1000}"', config)
        self.assertIn('EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"', config)


if __name__ == "__main__":
    unittest.main()
