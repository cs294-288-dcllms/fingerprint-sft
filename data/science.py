"""Local Mixture-of-Thoughts science dataset provider."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

from data.base import DatasetExample, DatasetProvider

BOXED_CHOICE_PATTERN = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)


def _dataset_path(split: str) -> Path:
    env_name = "SCIENCE_TRAIN_PATH" if split == "train" else "SCIENCE_EVAL_PATH"
    raw_path = os.environ.get(env_name)
    if not raw_path:
        raise RuntimeError(f"{env_name} must point to the prepared science JSONL file")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{env_name} does not exist: {path}")
    return path


def _extract_gold_choice(row: dict, assistant_text: str) -> str:
    metadata_choice = str(row.get("_meta", {}).get("gold_choice", "")).strip().upper()
    if metadata_choice in {"A", "B", "C", "D"}:
        return metadata_choice
    matches = BOXED_CHOICE_PATTERN.findall(assistant_text)
    if not matches:
        raise ValueError("science row is missing a boxed A-D answer")
    return matches[-1].upper()


class ScienceProvider(DatasetProvider):
    """Read prepared leakage-free science MCQ prompts from local JSONL."""

    name = "science"

    def load(self, split: str, limit: int | None = None) -> Iterable[DatasetExample]:
        path = _dataset_path(split)
        emitted = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    raise ValueError(f"{path}:{line_number} has invalid messages")
                user = messages[0]
                assistant = messages[-1]
                if user.get("role") != "user" or assistant.get("role") != "assistant":
                    raise ValueError(f"{path}:{line_number} is not a user/assistant example")
                prompt = str(user.get("content", "")).strip()
                assistant_text = str(assistant.get("content", "")).strip()
                if not prompt or not assistant_text:
                    raise ValueError(f"{path}:{line_number} has empty content")
                gold_choice = _extract_gold_choice(row, assistant_text)
                yield DatasetExample(prompt=prompt, solution=f"\\boxed{{{gold_choice}}}")
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


__all__ = ["ScienceProvider"]
