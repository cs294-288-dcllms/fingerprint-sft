#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def prompt_hash(row: dict[str, Any]) -> str | None:
    metadata = row.get("_meta")
    if isinstance(metadata, dict) and isinstance(metadata.get("prompt_sha256"), str):
        return metadata["prompt_sha256"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify two converted trace sets are paired.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()

    left_rows = read_rows(args.left)
    right_rows = read_rows(args.right)
    if len(left_rows) != len(right_rows):
        raise SystemExit(f"row-count mismatch: {len(left_rows)} != {len(right_rows)}")

    mismatches = []
    identical_responses = 0
    for index, (left, right) in enumerate(zip(left_rows, right_rows, strict=True)):
        if prompt_hash(left) != prompt_hash(right):
            mismatches.append(index)
        left_response = left.get("messages", [{}])[-1].get("content")
        right_response = right.get("messages", [{}])[-1].get("content")
        if left_response == right_response:
            identical_responses += 1

    if mismatches:
        preview = ", ".join(map(str, mismatches[:10]))
        raise SystemExit(f"{len(mismatches)} prompt mismatches; first indices: {preview}")

    print(
        json.dumps(
            {
                "paired_rows": len(left_rows),
                "prompt_mismatches": 0,
                "identical_responses": identical_responses,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
