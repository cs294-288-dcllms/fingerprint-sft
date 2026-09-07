"""Architecture-aware LoRA target selection."""

from __future__ import annotations

from models.loader import is_qwen35_config

DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

QWEN35_LINEAR_ATTENTION_TARGETS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def lora_target_modules(model) -> list[str]:
    """Return LoRA targets covering an architecture's language layers."""
    targets = list(DEFAULT_LORA_TARGETS)
    if is_qwen35_config(model.config):
        targets.extend(QWEN35_LINEAR_ATTENTION_TARGETS)
    return targets
