#!/usr/bin/env python3
"""Convert gold science thinking traces into Stage-3 prompt/response JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    return parser.parse_args()


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=True,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def strip_duplicate_think_prefix(prompt: str, response: str) -> str:
    stripped = response.lstrip()
    if prompt.rstrip().endswith("<think>") and stripped.startswith("<think>"):
        return stripped[len("<think>") :].lstrip("\n")
    return response


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{args.input}:{line_number}: invalid messages")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{args.input}:{line_number}: missing assistant target")
            prompt = render_prompt(tokenizer, messages[:-1])
            response = strip_duplicate_think_prefix(
                prompt, str(messages[-1].get("content", ""))
            )
            gold_choice = str(row.get("_meta", {}).get("gold_choice", "")).upper()
            if gold_choice not in {"A", "B", "C", "D"}:
                raise ValueError(f"{args.input}:{line_number}: invalid gold choice")
            payload = {
                "prompt": prompt,
                "response": response,
                "solution": f"\\boxed{{{gold_choice}}}",
            }
            json.dump(payload, destination, ensure_ascii=False)
            destination.write("\n")
            count += 1
    print(f"wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
