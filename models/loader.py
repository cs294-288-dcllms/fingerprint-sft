"""Model loading helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from config import ModelSpec
from packaging.version import Version
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedModel,
)
from transformers import (
    __version__ as transformers_version,
)

LOGGER = logging.getLogger(__name__)

_DTYPE_ALIASES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

_QWEN35_MULTIMODAL_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
_QWEN35_MIN_TRANSFORMERS = Version("5.8.0")


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve a string dtype alias to a torch.dtype.

    Args:
        name: Dtype alias (e.g., "bf16", "float16").

    Returns:
        torch.dtype corresponding to the alias.

    Raises:
        ValueError: If the dtype alias is unsupported.
    """
    key = name.lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype: {name}")
    return _DTYPE_ALIASES[key]


def is_qwen35_config(config) -> bool:
    """Return whether a model config belongs to the Qwen3.5 family."""
    return getattr(config, "model_type", "") in {
        *_QWEN35_MULTIMODAL_MODEL_TYPES,
        "qwen3_5_text",
        "qwen3_5_moe_text",
    }


def select_model_loader(config):
    """Choose an AutoModel loader compatible with a checkpoint config.

    Qwen3.5 checkpoints advertise a multimodal conditional-generation
    architecture, but ADFP only consumes text. Transformers 5.8+ can load the
    language backbone directly through AutoModelForCausalLM, skipping the
    unused vision tower while preserving generate(), KV caching, and logits.
    Other multimodal checkpoints fall back to AutoModelForImageTextToText.
    """
    model_type = getattr(config, "model_type", "")
    if model_type in _QWEN35_MULTIMODAL_MODEL_TYPES:
        installed = Version(transformers_version)
        if installed < _QWEN35_MIN_TRANSFORMERS:
            raise RuntimeError(
                "Qwen3.5 requires transformers>=5.8.0 for text-only causal-LM "
                f"loading; found {transformers_version}."
            )
        return AutoModelForCausalLM

    architectures = getattr(config, "architectures", None) or []
    has_vision_config = getattr(config, "vision_config", None) is not None
    is_conditional_generation = any(
        architecture.endswith("ForConditionalGeneration") for architecture in architectures
    )
    if has_vision_config and is_conditional_generation:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def load_causal_lm(
    spec: ModelSpec,
    adapter_path: Path | None = None,
) -> PreTrainedModel:
    """Load a causal LM in evaluation mode.

    Args:
        spec: ModelSpec with model name and dtype.
        adapter_path: Optional PEFT adapter to load on the base model.

    Returns:
        Text-generating model instance set to eval() mode.
    """
    dtype = resolve_dtype(spec.dtype)
    config = AutoConfig.from_pretrained(
        spec.name,
        trust_remote_code=True,
    )
    config.use_cache = True
    if getattr(config, "text_config", None) is not None:
        config.text_config.use_cache = True
    loader = select_model_loader(config)
    LOGGER.info(
        "Loading %s with %s (model_type=%s)",
        spec.name,
        loader.__name__,
        getattr(config, "model_type", "unknown"),
    )
    model = loader.from_pretrained(
        spec.name,
        config=config,
        dtype=dtype,
        trust_remote_code=True,
    )
    if adapter_path is not None:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            local_files_only=True,
        )
    model.config.use_cache = True
    model.eval()
    return model
