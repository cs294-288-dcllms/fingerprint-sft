#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATASET_ID = "open-r1/Mixture-of-Thoughts"
DATASET_CONFIG = "science"
DATASET_REVISION = "e55fa28006c0d0ec60fb3547520f775dd42d02cd"
MODEL_ID = "Qwen/Qwen3.5-4B"
BOXED_CHOICE_PATTERN = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-free Science SFT/eval splits containing complete thinking traces."
        )
    )
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=9_000)
    parser.add_argument("--eval-samples", type=int, default=1_000)
    parser.add_argument("--max-tokens", type=int, default=4_096)
    parser.add_argument("--max-traces-per-train-prompt", type=int, default=3)
    parser.add_argument(
        "--eval-bucket-denominator",
        type=int,
        default=5,
        help="One prompt-hash bucket out of this many is reserved for evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def stable_hash(*parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_choice(text: str) -> str | None:
    matches = BOXED_CHOICE_PATTERN.findall(text)
    return matches[-1].upper() if matches else None


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("expected exactly one user message and one assistant message")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message is not a mapping")
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if not role or not content:
            raise ValueError("message is missing role/content")
        normalized.append({"role": role, "content": content})
    if [message["role"] for message in normalized] != ["user", "assistant"]:
        raise ValueError("expected user/assistant roles")
    return normalized


def render_chatml(messages: list[dict[str, str]]) -> str:
    return "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages
    )


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(row["_meta"]["qwen35_tokens"]) for row in rows]
    sources = Counter(str(row["_meta"]["source"]) for row in rows)
    prompts = Counter(str(row["_meta"]["prompt_sha256"]) for row in rows)
    return {
        "samples": len(rows),
        "unique_prompts": len(prompts),
        "traces_per_prompt": dict(sorted(Counter(prompts.values()).items())),
        "source_counts": dict(sorted(sources.items())),
        "qwen35_tokens": {
            "min": min(lengths),
            "p50": percentile(lengths, 0.50),
            "p75": percentile(lengths, 0.75),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
    temporary.replace(path)
    return digest.hexdigest()


def shuffled_indices(length: int, seed: int) -> list[int]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices


def iter_rows(
    dataset: Iterable[dict[str, Any]],
    seed: int,
    streaming: bool,
) -> Iterable[tuple[int, dict[str, Any]]]:
    if streaming:
        for scan_index, row in enumerate(dataset.shuffle(seed=seed, buffer_size=50_000)):
            yield scan_index, dict(row)
        return

    for source_index in shuffled_indices(len(dataset), seed):
        yield source_index, dict(dataset[source_index])


def main() -> None:
    args = parse_args()
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise SystemExit("train/eval sample counts must be positive")
    if args.eval_bucket_denominator < 2:
        raise SystemExit("--eval-bucket-denominator must be at least 2")
    if args.max_traces_per_train_prompt <= 0:
        raise SystemExit("--max-traces-per-train-prompt must be positive")

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install the pinned SFT environment before preparing data.") from exc

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
        revision=DATASET_REVISION,
        streaming=args.streaming,
    )

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    train_prompt_counts: defaultdict[str, int] = defaultdict(int)
    eval_prompts: set[str] = set()
    seen_traces: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    scanned = 0

    for source_index, row in iter_rows(dataset, args.seed, args.streaming):
        scanned += 1
        try:
            messages = normalize_messages(row.get("messages"))
        except ValueError:
            rejection_counts["invalid_messages"] += 1
            continue

        user_text = messages[0]["content"]
        assistant_text = messages[1]["content"]
        if "<think>" not in assistant_text or "</think>" not in assistant_text:
            rejection_counts["incomplete_thinking_trace"] += 1
            continue
        gold_choice = extract_choice(assistant_text)
        if gold_choice is None:
            rejection_counts["missing_boxed_choice"] += 1
            continue

        prompt_sha256 = stable_hash(user_text)
        response_sha256 = stable_hash(assistant_text)
        trace_sha256 = stable_hash(prompt_sha256, response_sha256)
        if trace_sha256 in seen_traces:
            rejection_counts["duplicate_trace"] += 1
            continue

        qwen35_tokens = len(
            tokenizer(render_chatml(messages), add_special_tokens=False)["input_ids"]
        )
        if qwen35_tokens > args.max_tokens:
            rejection_counts["over_token_limit"] += 1
            continue

        bucket = int(stable_hash(str(args.seed), user_text)[:16], 16)
        is_eval = bucket % args.eval_bucket_denominator == 0
        if is_eval:
            if len(eval_rows) >= args.eval_samples:
                rejection_counts["eval_full"] += 1
                continue
            if prompt_sha256 in eval_prompts:
                rejection_counts["extra_eval_trace"] += 1
                continue
        else:
            if len(train_rows) >= args.train_samples:
                rejection_counts["train_full"] += 1
                continue
            if train_prompt_counts[prompt_sha256] >= args.max_traces_per_train_prompt:
                rejection_counts["too_many_train_traces"] += 1
                continue

        record = {
            "messages": messages,
            "_meta": {
                "dataset": DATASET_ID,
                "config": DATASET_CONFIG,
                "revision": DATASET_REVISION,
                "source": str(row.get("source", "")),
                "source_index": source_index,
                "source_num_tokens": int(row.get("num_tokens") or 0),
                "qwen35_tokens": qwen35_tokens,
                "prompt_sha256": prompt_sha256,
                "response_sha256": response_sha256,
                "gold_choice": gold_choice,
            },
        }
        seen_traces.add(trace_sha256)
        if is_eval:
            eval_rows.append(record)
            eval_prompts.add(prompt_sha256)
        else:
            train_rows.append(record)
            train_prompt_counts[prompt_sha256] += 1

        if len(train_rows) >= args.train_samples and len(eval_rows) >= args.eval_samples:
            break

    if len(train_rows) != args.train_samples or len(eval_rows) != args.eval_samples:
        raise RuntimeError(
            f"insufficient eligible rows after scanning {scanned}: "
            f"train={len(train_rows)}/{args.train_samples}, "
            f"eval={len(eval_rows)}/{args.eval_samples}"
        )

    train_prompt_set = {str(row["_meta"]["prompt_sha256"]) for row in train_rows}
    eval_prompt_set = {str(row["_meta"]["prompt_sha256"]) for row in eval_rows}
    overlap = train_prompt_set & eval_prompt_set
    if overlap:
        raise RuntimeError(f"prompt leakage detected across splits: {len(overlap)} prompts")

    def order_key(split: str, row: dict[str, Any]) -> str:
        return stable_hash(
            str(args.seed),
            split,
            str(row["_meta"]["prompt_sha256"]),
            str(row["_meta"]["response_sha256"]),
        )

    train_rows.sort(key=lambda row: order_key("train", row))
    eval_rows.sort(key=lambda row: order_key("eval", row))
    train_sha256 = write_jsonl(args.train_output, train_rows)
    eval_sha256 = write_jsonl(args.eval_output, eval_rows)

    metadata = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "model_tokenizer": MODEL_ID,
        "seed": args.seed,
        "selection_order": "deterministic_global_shuffle",
        "max_tokens": args.max_tokens,
        "max_traces_per_train_prompt": args.max_traces_per_train_prompt,
        "eval_bucket_denominator": args.eval_bucket_denominator,
        "source_rows_scanned": scanned,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "prompt_overlap": 0,
        "train": {
            **split_stats(train_rows),
            "path": str(args.train_output),
            "sha256": train_sha256,
        },
        "eval": {
            **split_stats(eval_rows),
            "path": str(args.eval_output),
            "sha256": eval_sha256,
        },
    }
    metadata_path = args.train_output.parent / "science_thinking_9k_1k.meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
