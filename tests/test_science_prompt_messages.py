from __future__ import annotations

import unittest

from models.prompts import PromptBuilder


class ScienceMessagePromptTest(unittest.TestCase):
    def test_message_prompt_uses_configured_thinking_mode(self) -> None:
        class FakeTokenizer:
            def __init__(self) -> None:
                self.kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        tokenizer = FakeTokenizer()
        rendered = PromptBuilder(enable_thinking=True).build_from_messages(
            tokenizer,
            [{"role": "user", "content": "Science question?"}],
        )
        self.assertEqual(rendered, "rendered")
        self.assertIs(tokenizer.kwargs["enable_thinking"], True)

    def test_user_only_science_prompt_matches_evaluation_shape(self) -> None:
        class FakeTokenizer:
            def __init__(self) -> None:
                self.messages = None
                self.kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "rendered"

        tokenizer = FakeTokenizer()
        rendered = PromptBuilder(system_prompt=None, enable_thinking=True).build(
            tokenizer, "Science question?"
        )
        self.assertEqual(rendered, "rendered")
        self.assertEqual(
            tokenizer.messages,
            [{"role": "user", "content": "Science question?\n"}],
        )
        self.assertIs(tokenizer.kwargs["enable_thinking"], True)


if __name__ == "__main__":
    unittest.main()
