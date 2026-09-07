from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from config import ModelSpec
from models.loader import load_causal_lm


class LoaderRuntimeTest(unittest.TestCase):
    def test_use_cache_is_set_on_config_not_forwarded_to_constructor(self) -> None:
        config = SimpleNamespace(
            model_type="text_model",
            architectures=["TextForCausalLM"],
            vision_config=None,
            use_cache=False,
        )
        model = SimpleNamespace(
            config=SimpleNamespace(use_cache=False),
            eval=lambda: None,
        )

        with (
            patch("models.loader.AutoConfig.from_pretrained", return_value=config),
            patch(
                "models.loader.AutoModelForCausalLM.from_pretrained", return_value=model
            ) as from_pretrained,
        ):
            result = load_causal_lm(ModelSpec(name="test/model", dtype="bfloat16"))

        self.assertIs(result, model)
        self.assertIs(config.use_cache, True)
        self.assertIs(model.config.use_cache, True)
        self.assertNotIn("use_cache", from_pretrained.call_args.kwargs)
        self.assertIs(from_pretrained.call_args.kwargs["dtype"], torch.bfloat16)

    def test_optional_adapter_wraps_loaded_base_model(self) -> None:
        config = SimpleNamespace(
            model_type="text_model",
            architectures=["TextForCausalLM"],
            vision_config=None,
            use_cache=False,
        )
        base_model = SimpleNamespace(
            config=SimpleNamespace(use_cache=False),
            eval=lambda: None,
        )
        wrapped_model = SimpleNamespace(
            config=SimpleNamespace(use_cache=False),
            eval=lambda: None,
        )

        with tempfile.TemporaryDirectory() as directory:
            adapter_path = Path(directory)
            with (
                patch("models.loader.AutoConfig.from_pretrained", return_value=config),
                patch(
                    "models.loader.AutoModelForCausalLM.from_pretrained",
                    return_value=base_model,
                ),
                patch(
                    "peft.PeftModel.from_pretrained",
                    return_value=wrapped_model,
                ) as peft_from_pretrained,
            ):
                result = load_causal_lm(
                    ModelSpec(name="test/model", dtype="bfloat16"),
                    adapter_path,
                )

        self.assertIs(result, wrapped_model)
        peft_from_pretrained.assert_called_once_with(
            base_model,
            str(adapter_path),
            local_files_only=True,
        )
        self.assertIs(wrapped_model.config.use_cache, True)


if __name__ == "__main__":
    unittest.main()
