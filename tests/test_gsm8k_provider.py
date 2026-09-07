from __future__ import annotations

import unittest
from unittest.mock import patch

from data.gsm8k import GSM8KProvider


class GSM8KProviderTest(unittest.TestCase):
    def test_uses_current_huggingface_dataset_id(self) -> None:
        rows = [{"question": "2 + 2?", "answer": "#### 4"}]
        with patch("data.gsm8k.load_dataset", return_value=rows) as load_dataset:
            examples = list(GSM8KProvider().load("train", limit=1))

        load_dataset.assert_called_once_with("openai/gsm8k", "main", split="train")
        self.assertEqual(examples[0].prompt, "2 + 2?")
        self.assertEqual(examples[0].solution, "#### 4")


if __name__ == "__main__":
    unittest.main()
