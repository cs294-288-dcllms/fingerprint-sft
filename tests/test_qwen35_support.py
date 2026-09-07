from __future__ import annotations

import unittest
from types import SimpleNamespace

from models.loader import select_model_loader
from models.lora import lora_target_modules
from models.prompts import PromptBuilder
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText


class Qwen35LoaderTest(unittest.TestCase):
    def test_multimodal_qwen35_uses_text_only_causal_loader(self) -> None:
        config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            vision_config=object(),
        )
        self.assertIs(select_model_loader(config), AutoModelForCausalLM)

    def test_other_multimodal_conditional_model_uses_image_text_loader(self) -> None:
        config = SimpleNamespace(
            model_type="other_vlm",
            architectures=["OtherForConditionalGeneration"],
            vision_config=object(),
        )
        self.assertIs(select_model_loader(config), AutoModelForImageTextToText)


class Qwen35LoraTest(unittest.TestCase):
    def test_linear_attention_projections_are_targeted(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_text"))
        targets = lora_target_modules(model)
        self.assertIn("q_proj", targets)
        self.assertIn("in_proj_qkv", targets)
        self.assertIn("out_proj", targets)


class PromptBuilderTest(unittest.TestCase):
    def test_qwen_thinking_is_disabled_when_supported(self) -> None:
        class FakeTokenizer:
            def __init__(self) -> None:
                self.kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        tokenizer = FakeTokenizer()
        rendered = PromptBuilder().build(tokenizer, "2 + 2?")
        self.assertEqual(rendered, "rendered")
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_qwen_thinking_can_be_enabled_for_science(self) -> None:
        class FakeTokenizer:
            def __init__(self) -> None:
                self.kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        tokenizer = FakeTokenizer()
        rendered = PromptBuilder(enable_thinking=True).build(tokenizer, "Science question?")
        self.assertEqual(rendered, "rendered")
        self.assertIs(tokenizer.kwargs["enable_thinking"], True)


if __name__ == "__main__":
    unittest.main()
