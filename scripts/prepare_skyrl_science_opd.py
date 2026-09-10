#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

BOXED_CHOICE = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the local science MCQ JSONL splits to SkyRL parquet."
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path, split: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_number} has invalid messages")
            prompt = messages[:-1]
            assistant = str(messages[-1].get("content", ""))
            gold_choice = row.get("_meta", {}).get("gold_choice")
            if not gold_choice:
                matches = BOXED_CHOICE.findall(assistant)
                gold_choice = matches[-1].upper() if matches else None
            if gold_choice not in {"A", "B", "C", "D"}:
                raise ValueError(f"{path}:{line_number} has no valid gold choice")
            converted.append(
                {
                    "data_source": "open-r1/Mixture-of-Thoughts:science",
                    "prompt": prompt,
                    "env_class": "aime",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": gold_choice,
                    },
                    "extra_info": {
                        "split": split,
                        "index": len(converted),
                        "gold_choice": gold_choice,
                        "prompt_sha256": row.get("_meta", {}).get("prompt_sha256"),
                    },
                }
            )
    return converted


def write_parquet(rows: list[dict[str, Any]], output: Path) -> None:
    from datasets import Dataset

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Dataset.from_list(rows).to_parquet(str(temporary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    train_rows = read_rows(args.train_jsonl, "train")
    eval_rows = read_rows(args.eval_jsonl, "eval")
    if len(train_rows) != 9_000:
        raise ValueError(f"Expected 9000 training rows, found {len(train_rows)}")
    if len(eval_rows) != 1_000:
        raise ValueError(f"Expected 1000 evaluation rows, found {len(eval_rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(train_rows, args.output_dir / "train.parquet")
    write_parquet(eval_rows, args.output_dir / "validation.parquet")
    manifest = {
        "train_source": str(args.train_jsonl.resolve()),
        "eval_source": str(args.eval_jsonl.resolve()),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "reward_note": "Environment rewards are replaced by teacher-token KL in OPD.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
