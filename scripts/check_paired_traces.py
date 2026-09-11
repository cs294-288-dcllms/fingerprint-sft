#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def prompt_hash(row: dict[str, Any]) -> str:
    metadata = row.get("_meta")
    if isinstance(metadata, dict) and isinstance(metadata.get("prompt_sha256"), str):
        return metadata["prompt_sha256"]
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    raise ValueError("trace row has no prompt identity")


def response_text(row: dict[str, Any]) -> str | None:
    response = row.get("response")
    if isinstance(response, str):
        return response
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                return content if isinstance(content, str) else None
    return None


def solution_identity(row: dict[str, Any]) -> str | None:
    solution = row.get("solution")
    if isinstance(solution, str):
        return solution
    metadata = row.get("_meta")
    if isinstance(metadata, dict):
        choice = metadata.get("gold_choice")
        return choice if isinstance(choice, str) else None
    return None


def metadata_seed(path: Path) -> int:
    metadata_path = path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that two trace sets use the same prompts with different teacher samples."
        )
    )
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--right-is-prefix", action="store_true")
    parser.add_argument("--expected-left-seed", type=int)
    parser.add_argument("--expected-right-seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left_rows = read_rows(args.left)
    right_rows = read_rows(args.right)
    if args.right_is_prefix:
        if not right_rows or len(left_rows) < len(right_rows):
            raise SystemExit(
                f"invalid prefix sizes: left={len(left_rows)}, right={len(right_rows)}"
            )
        left_rows = left_rows[: len(right_rows)]
    elif len(left_rows) != len(right_rows):
        raise SystemExit(f"row-count mismatch: {len(left_rows)} != {len(right_rows)}")

    prompt_mismatches: list[int] = []
    solution_mismatches: list[int] = []
    identical_responses = 0
    prompt_digests: list[str] = []
    for index, (left, right) in enumerate(zip(left_rows, right_rows, strict=True)):
        left_prompt = prompt_hash(left)
        right_prompt = prompt_hash(right)
        prompt_digests.append(right_prompt)
        if left_prompt != right_prompt:
            prompt_mismatches.append(index)
        left_solution = solution_identity(left)
        right_solution = solution_identity(right)
        if (
            left_solution is not None
            and right_solution is not None
            and left_solution != right_solution
        ):
            solution_mismatches.append(index)
        if response_text(left) == response_text(right):
            identical_responses += 1

    if prompt_mismatches:
        preview = ", ".join(map(str, prompt_mismatches[:10]))
        raise SystemExit(f"{len(prompt_mismatches)} prompt mismatches; first indices: {preview}")
    if solution_mismatches:
        preview = ", ".join(map(str, solution_mismatches[:10]))
        raise SystemExit(
            f"{len(solution_mismatches)} solution mismatches; first indices: {preview}"
        )
    if identical_responses == len(right_rows):
        raise SystemExit("teacher responses are identical across the two trace sets")

    left_seed = metadata_seed(args.left) if args.expected_left_seed is not None else None
    right_seed = metadata_seed(args.right) if args.expected_right_seed is not None else None
    if args.expected_left_seed is not None and left_seed != args.expected_left_seed:
        raise SystemExit(f"left seed mismatch: {left_seed} != {args.expected_left_seed}")
    if args.expected_right_seed is not None and right_seed != args.expected_right_seed:
        raise SystemExit(f"right seed mismatch: {right_seed} != {args.expected_right_seed}")

    report = {
        "paired_prompts": len(right_rows),
        "prompt_mismatches": 0,
        "solution_mismatches": 0,
        "identical_teacher_responses": identical_responses,
        "different_teacher_responses": len(right_rows) - identical_responses,
        "baseline_teacher_seed": left_seed,
        "resampled_teacher_seed": right_seed,
        "prompt_sha256": hashlib.sha256("\0".join(prompt_digests).encode("utf-8")).hexdigest(),
        "baseline_trace_file": str(args.left.resolve()),
        "resampled_trace_file": str(args.right.resolve()),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
