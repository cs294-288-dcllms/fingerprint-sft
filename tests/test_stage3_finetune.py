import unittest

from stages.stage3_finetune import _mask_prompt


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
    def test_appends_eos_to_response_labels(self):
        row = _mask_prompt(FakeTokenizer(), "prompt", "response", 16)
        self.assertEqual(row["input_ids"], [1, 2, 3, 4, 9])
        self.assertEqual(row["labels"], [-100, -100, 3, 4, 9])

    def test_does_not_duplicate_existing_eos(self):
        row = _mask_prompt(FakeTokenizer(), "prompt", "ended", 16)
        self.assertEqual(row["input_ids"], [1, 2, 3, 9])
        self.assertEqual(row["labels"], [-100, -100, 3, 9])


if __name__ == "__main__":
    unittest.main()
