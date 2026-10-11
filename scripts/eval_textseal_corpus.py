#!/usr/bin/env python3
"""Evaluate direct TextSeal detectability and Science MCQ accuracy for a corpus."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)
ANSWER_PATTERN = re.compile(
    r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*[*\\(\[]*([A-D])\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct TextSeal detection on teacher traces or student generations."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--text-key", default="response")
    parser.add_argument("--solution-key", default="solution")
    parser.add_argument("--secret-key", type=int, default=294288)
    parser.add_argument("--secret-key-b", type=int, default=None)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--mixing-alpha", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--score-chunk", type=int, default=131072)
    return parser.parse_args()


def extract_choice(text: str) -> str | None:
    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        return boxed[-1].upper()
    answers = ANSWER_PATTERN.findall(text)
    return answers[-1].upper() if answers else None


def _read_jsonl(path: Path, *, skip_partial_tail: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if skip_partial_tail and not line.endswith("\n"):
                continue
            rows.append(json.loads(line))
    return rows


def read_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    if path.is_file():
        rows = _read_jsonl(path, skip_partial_tail=False)
        return rows[:max_samples] if max_samples > 0 else rows
    if not path.is_dir():
        raise FileNotFoundError(path)

    shard_paths = sorted(path.glob("rank_*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"no rank_*.jsonl shards in {path}")
    rows = [
        row
        for shard_path in shard_paths
        for row in _read_jsonl(shard_path, skip_partial_tail=True)
    ]
    if rows and all(isinstance(row.get("index"), int) for row in rows):
        rows.sort(key=lambda row: row["index"])
        indices = [row["index"] for row in rows]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate sample indices in {path}")
    return rows[:max_samples] if max_samples > 0 else rows


def unique_pairs(token_ids: list[int], ngram: int) -> tuple[np.ndarray, np.ndarray]:
    """Return TextSeal v2 pairs using the official detector's start position."""
    if len(token_ids) <= ngram + 1:
        return (
            np.empty((0, ngram), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
    tokens = np.asarray(token_ids, dtype=np.int32)
    contexts = np.lib.stride_tricks.sliding_window_view(tokens, ngram)[1:-1].copy()
    targets = tokens[ngram + 1 :].copy()
    keys = np.concatenate([contexts, targets[:, None]], axis=1)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    first_indices.sort()
    return contexts[first_indices], targets[first_indices]


def score_pairs(
    contexts: np.ndarray,
    targets: np.ndarray,
    *,
    spec: Any,
    chunk_size: int,
) -> np.ndarray:
    import torch

    from models.textseal import textseal_fused_scores

    chunks: list[np.ndarray] = []
    for start in range(0, targets.size, chunk_size):
        stop = start + chunk_size
        chunk_contexts = torch.from_numpy(contexts[start:stop].astype(np.int64))
        chunk_targets = torch.from_numpy(targets[start:stop].astype(np.int64))
        chunks.append(
            textseal_fused_scores(
                chunk_contexts,
                chunk_targets,
                spec,
            )
            .numpy()
            .astype(np.float32)
        )
    return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.float32)


def unweighted_gamma(scores: np.ndarray, base_variance: float) -> dict[str, Any]:
    from scipy import special

    count = int(scores.size)
    if count == 0:
        return {
            "count": 0,
            "mean_score": 0.0,
            "p_value": 1.0,
            "negative_log10_p": 0.0,
            "detected_at_0_01": False,
        }
    score_sum = float(scores.astype(np.float64).sum())
    p_value = float(
        special.gammaincc(
            count / base_variance,
            score_sum / base_variance,
        )
    )
    return {
        "count": count,
        "mean_score": score_sum / count,
        "p_value": p_value,
        "negative_log10_p": -math.log10(max(p_value, 1e-300)),
        "detected_at_0_01": p_value < 0.01,
    }


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile)) if values else 0.0


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    from transformers import AutoTokenizer

    from models.textseal import TextSealSpec

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )
    secret_key_b = (
        args.secret_key_b if args.secret_key_b is not None else args.secret_key + 12345
    )
    spec = TextSealSpec(
        secret_key=args.secret_key,
        secret_key_b=secret_key_b,
        ngram=args.ngram,
        mixing_alpha=args.mixing_alpha,
        seed=0,
    )
    rows = read_rows(args.input, args.max_samples)
    powers: list[float] = []
    scored_counts: list[int] = []
    detected = 0
    parsed = 0
    correct = 0
    evaluable = 0
    think_closed = 0
    corpus_contexts: list[np.ndarray] = []
    corpus_targets: list[np.ndarray] = []
    corpus_scores: list[np.ndarray] = []
    base_variance = args.mixing_alpha**2 + (1.0 - args.mixing_alpha) ** 2

    for index, row in enumerate(rows):
        text = row.get(args.text_key)
        if not isinstance(text, str):
            raise ValueError(f"row {index} has no string {args.text_key!r}")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        contexts, targets = unique_pairs(token_ids, args.ngram)
        scores = score_pairs(
            contexts,
            targets,
            spec=spec,
            chunk_size=args.score_chunk,
        )
        trace_result = unweighted_gamma(scores, base_variance)
        powers.append(float(trace_result["negative_log10_p"]))
        scored_counts.append(int(trace_result["count"]))
        detected += int(trace_result["detected_at_0_01"])
        corpus_contexts.append(contexts)
        corpus_targets.append(targets)
        corpus_scores.append(scores)
        think_closed += int("</think>" in text)

        solution = row.get(args.solution_key)
        if isinstance(solution, str):
            evaluable += 1
            prediction_choice = extract_choice(text)
            solution_choice = extract_choice(solution)
            parsed += int(prediction_choice is not None)
            correct += int(
                prediction_choice is not None
                and solution_choice is not None
                and prediction_choice == solution_choice
            )

    all_contexts = (
        np.concatenate(corpus_contexts, axis=0)
        if corpus_contexts
        else np.empty((0, args.ngram), dtype=np.int32)
    )
    all_targets = (
        np.concatenate(corpus_targets)
        if corpus_targets
        else np.empty((0,), dtype=np.int32)
    )
    all_scores = (
        np.concatenate(corpus_scores)
        if corpus_scores
        else np.empty((0,), dtype=np.float32)
    )
    all_keys = np.concatenate([all_contexts, all_targets[:, None]], axis=1)
    if all_keys.shape[0]:
        _, first_indices = np.unique(all_keys, axis=0, return_index=True)
        first_indices.sort()
    else:
        first_indices = np.empty((0,), dtype=np.int64)
    pooled = unweighted_gamma(all_scores[first_indices], base_variance)

    summary = {
        "input": str(args.input.resolve()),
        "text_key": args.text_key,
        "tokenizer": args.tokenizer,
        "samples": len(rows),
        "textseal": {
            "secret_key": args.secret_key,
            "secret_key_b": secret_key_b,
            "ngram": args.ngram,
            "mixing_alpha": args.mixing_alpha,
        },
        "direct_detection": {
            "detected_at_0_01": detected,
            "detection_rate_at_0_01": detected / len(rows) if rows else 0.0,
            "median_negative_log10_p": percentile(powers, 0.5),
            "mean_negative_log10_p": float(np.mean(powers)) if powers else 0.0,
            "p25_negative_log10_p": percentile(powers, 0.25),
            "p75_negative_log10_p": percentile(powers, 0.75),
            "median_scored_tokens": percentile(
                [float(value) for value in scored_counts],
                0.5,
            ),
            "pooled_unique_pairs_before_global_dedup": int(all_keys.shape[0]),
            "pooled_unique_pairs_after_global_dedup": int(first_indices.size),
            "pooled": pooled,
        },
        "trace_quality": {
            "think_closed": think_closed,
            "think_close_rate": think_closed / len(rows) if rows else 0.0,
            "evaluable": evaluable,
            "parsed": parsed,
            "parse_rate": parsed / evaluable if evaluable else None,
            "correct": correct,
            "accuracy": correct / evaluable if evaluable else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
