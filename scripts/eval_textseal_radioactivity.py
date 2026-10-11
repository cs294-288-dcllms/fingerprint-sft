#!/usr/bin/env python3
"""Paper-style open-model TextSeal radioactivity evaluation.

The primary statistic globally deduplicates n-gram contexts (TextSeal ``v1``).
The output also reports global context-plus-prediction deduplication (``v2``)
from the same teacher-forced forward pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class TraceSet:
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-force traces through a student, score top-1 predictions "
            "with TextSeal, and globally deduplicate (context, prediction) pairs."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--trace-set",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for known and independently sampled trace sets.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--secret-key", type=int, default=294288)
    parser.add_argument("--secret-key-b", type=int, default=None)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--mixing-alpha", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-traces", type=int, default=8)
    parser.add_argument("--entropy-chunk", type=int, default=128)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    return parser.parse_args()


def parse_trace_sets(values: list[str]) -> list[TraceSet]:
    trace_sets: list[TraceSet] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"trace set must be LABEL=PATH, received {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser().resolve()
        if not label or label in labels:
            raise ValueError(f"trace-set label is empty or duplicated: {label!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        labels.add(label)
        trace_sets.append(TraceSet(label=label, path=path))
    return trace_sets


def configure_runtime_caches() -> None:
    runtime_root = Path(
        os.environ.get("RUNTIME_CACHE_ROOT")
        or Path(os.environ.get("XDG_CACHE_HOME") or "/tmp") / "fingerprint-sft"
    )
    defaults = {
        "TRITON_CACHE_DIR": runtime_root / "triton-textseal-radioactivity",
        "TORCHINDUCTOR_CACHE_DIR": runtime_root / "torchinductor-textseal-radioactivity",
        "CUDA_CACHE_PATH": runtime_root / "cuda",
    }
    for variable, default in defaults.items():
        path = Path(os.environ.setdefault(variable, str(default)))
        path.mkdir(parents=True, exist_ok=True)


def read_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("response"), str) or not row["response"].strip():
                raise ValueError(f"{path}:{line_number} has no non-empty response")
            if not row.get("prompt") and not row.get("messages"):
                raise ValueError(f"{path}:{line_number} has no prompt/messages")
            rows.append(row)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def prompt_text(row: dict[str, Any], tokenizer: Any) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("trace row has invalid messages")
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_model(
    model_name: str,
    adapter: Path | None,
    dtype_name: str,
    local_rank: int,
) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    config.use_cache = False
    if getattr(config, "text_config", None) is not None:
        config.text_config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    if adapter is not None:
        if not adapter.is_dir():
            raise FileNotFoundError(adapter)
        model = PeftModel.from_pretrained(
            model,
            str(adapter),
            local_files_only=True,
        )
    model.to(torch.device(f"cuda:{local_rank}"))
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, tokenizer


def encode_trace(
    row: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[list[int], int]:
    prompt_ids = tokenizer(
        prompt_text(row, tokenizer),
        add_special_tokens=False,
    )["input_ids"]
    response_ids = tokenizer(
        row["response"],
        add_special_tokens=False,
    )["input_ids"]
    if tokenizer.eos_token_id is not None and (
        not response_ids or response_ids[-1] != tokenizer.eos_token_id
    ):
        response_ids.append(tokenizer.eos_token_id)
    combined = (prompt_ids + response_ids)[:max_seq_length]
    return combined, min(len(prompt_ids), len(combined))


def response_positions(
    token_ids: list[int],
    response_start: int,
    ngram: int,
) -> list[int]:
    """Return every response-token position with a complete n-gram context."""
    return list(range(max(response_start, ngram), len(token_ids)))


def entropy_weights(entropies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if entropies.size == 0:
        return entropies.copy(), entropies.copy()
    minimum = float(entropies.min())
    maximum = float(entropies.max())
    if maximum - minimum < 1e-8:
        normalized = np.ones_like(entropies, dtype=np.float32)
    else:
        normalized = ((entropies - minimum) / (maximum - minimum)).astype(np.float32)
    sqrt_weights = np.sqrt(normalized).astype(np.float32)
    linear_weights = (0.1 + 0.9 * normalized).astype(np.float32)
    return sqrt_weights, linear_weights


def score_trace_logits(
    token_ids: list[int],
    response_start: int,
    logits: Any,
    *,
    spec: Any,
    entropy_chunk: int,
) -> dict[str, np.ndarray]:
    import torch

    from models.textseal import textseal_fused_scores

    positions = response_positions(token_ids, response_start, spec.ngram)
    if not positions:
        return {
            "contexts": np.empty((0, spec.ngram), dtype=np.int32),
            "predictions": np.empty((0,), dtype=np.int32),
            "scores": np.empty((0,), dtype=np.float32),
            "entropies": np.empty((0,), dtype=np.float32),
            "weights_sqrt": np.empty((0,), dtype=np.float32),
            "weights_linear": np.empty((0,), dtype=np.float32),
        }

    prediction_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []
    device = logits.device
    with torch.inference_mode():
        for start in range(0, len(positions), entropy_chunk):
            chunk_positions = positions[start : start + entropy_chunk]
            prediction_logits = logits[
                torch.tensor(
                    [position - 1 for position in chunk_positions],
                    device=device,
                )
            ].float()
            predictions = torch.argmax(prediction_logits, dim=-1)
            log_normalizer = torch.logsumexp(prediction_logits, dim=-1)
            probabilities = torch.softmax(prediction_logits, dim=-1)
            entropies = log_normalizer - (probabilities * prediction_logits).sum(dim=-1)
            prediction_chunks.append(predictions.cpu())
            entropy_chunks.append(entropies.cpu())
    del logits

    predictions = torch.cat(prediction_chunks)
    entropies = torch.cat(entropy_chunks).numpy().astype(np.float32)
    contexts = torch.tensor(
        [token_ids[position - spec.ngram : position] for position in positions],
        dtype=torch.long,
    )
    scores = textseal_fused_scores(contexts, predictions, spec).numpy().astype(np.float32)
    weights_sqrt, weights_linear = entropy_weights(entropies)
    return {
        "contexts": contexts.numpy().astype(np.int32),
        "predictions": predictions.numpy().astype(np.int32),
        "scores": scores,
        "entropies": entropies,
        "weights_sqrt": weights_sqrt,
        "weights_linear": weights_linear,
    }


def process_traces(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    spec: Any,
    max_seq_length: int,
    entropy_chunk: int,
) -> list[dict[str, np.ndarray]]:
    """Teacher-force a right-padded batch without changing real-token positions."""
    import torch

    encoded = [encode_trace(row, tokenizer, max_seq_length) for row in rows]
    if not encoded:
        return []

    device = next(model.parameters()).device
    maximum_length = max(len(token_ids) for token_ids, _ in encoded)
    input_tensor = torch.full(
        (len(encoded), maximum_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    for batch_index, (token_ids, _) in enumerate(encoded):
        trace_length = len(token_ids)
        input_tensor[batch_index, :trace_length] = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=device,
        )

    # Padding is strictly to the right, so causal real-token logits cannot see it.
    with torch.inference_mode():
        batch_logits = model(
            input_ids=input_tensor,
            use_cache=False,
        ).logits

    results = [
        score_trace_logits(
            token_ids,
            response_start,
            batch_logits[batch_index, : len(token_ids)],
            spec=spec,
            entropy_chunk=entropy_chunk,
        )
        for batch_index, (token_ids, response_start) in enumerate(encoded)
    ]
    del batch_logits
    return results


def process_trace(
    row: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    spec: Any,
    max_seq_length: int,
    entropy_chunk: int,
) -> dict[str, np.ndarray]:
    """Backward-compatible single-trace wrapper."""
    return process_traces(
        [row],
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        max_seq_length=max_seq_length,
        entropy_chunk=entropy_chunk,
    )[0]


def completed_indices(work_dir: Path) -> set[int]:
    completed: set[int] = set()
    for part_path in sorted(work_dir.glob("part_*.npz")):
        with np.load(part_path) as part:
            completed.update(int(value) for value in part["completed_indices"])
    return completed


def save_part(
    path: Path,
    completed: list[int],
    arrays: list[dict[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arrays:
        payload = {
            key: np.concatenate([array[key] for array in arrays], axis=0)
            for key in arrays[0]
        }
    else:
        payload = {
            "contexts": np.empty((0, 3), dtype=np.int32),
            "predictions": np.empty((0,), dtype=np.int32),
            "scores": np.empty((0,), dtype=np.float32),
            "entropies": np.empty((0,), dtype=np.float32),
            "weights_sqrt": np.empty((0,), dtype=np.float32),
            "weights_linear": np.empty((0,), dtype=np.float32),
        }
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        completed_indices=np.asarray(completed, dtype=np.int32),
        **payload,
    )
    temporary.replace(path)


def gamma_summary(
    scores: np.ndarray,
    weights: np.ndarray,
    base_variance: float,
) -> dict[str, float | int | bool]:
    from scipy import special

    count = int(scores.size)
    if count == 0:
        return {
            "count": 0,
            "score_sum": 0.0,
            "weight_sum": 0.0,
            "weighted_mean_score": 0.0,
            "p_value": 1.0,
            "negative_log10_p": 0.0,
            "detected_at_0_01": False,
        }
    weights64 = weights.astype(np.float64)
    scores64 = scores.astype(np.float64)
    weight_sum = float(weights64.sum())
    weighted_score = float((weights64 * scores64).sum())
    variance = float(base_variance * np.square(weights64).sum())
    if weight_sum <= 0.0 or variance <= 0.0:
        p_value = 1.0
    else:
        shape = weight_sum * weight_sum / variance
        scale = variance / weight_sum
        p_value = float(special.gammaincc(shape, weighted_score / scale))
    floored = max(p_value, 1e-300)
    return {
        "count": count,
        "score_sum": weighted_score,
        "weight_sum": weight_sum,
        "weighted_mean_score": weighted_score / weight_sum if weight_sum else 0.0,
        "p_value": p_value,
        "negative_log10_p": -math.log10(floored),
        "detected_at_0_01": p_value < 0.01,
    }


def metric_family(
    scores: np.ndarray,
    sqrt_weights: np.ndarray,
    linear_weights: np.ndarray,
    indices: np.ndarray,
    base_variance: float,
) -> dict[str, Any]:
    """Summarize all entropy-weighting variants for selected positions."""
    selected_scores = scores[indices]
    return {
        "unweighted": gamma_summary(
            selected_scores,
            np.ones_like(selected_scores, dtype=np.float32),
            base_variance,
        ),
        "sqrt_normalized_entropy": gamma_summary(
            selected_scores,
            sqrt_weights[indices],
            base_variance,
        ),
        "linear_normalized_entropy": gamma_summary(
            selected_scores,
            linear_weights[indices],
            base_variance,
        ),
    }


def first_unique_indices(keys: np.ndarray) -> np.ndarray:
    """Return stable first-occurrence indices for row-wise keys."""
    if not keys.shape[0]:
        return np.empty((0,), dtype=np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    indices.sort()
    return indices


def aggregate_trace_set(
    trace_set: TraceSet,
    *,
    output_dir: Path,
    expected_indices: set[int],
    spec: Any,
) -> dict[str, Any]:
    work_root = output_dir / ".work" / trace_set.label
    part_paths = sorted(work_root.glob("rank_*/part_*.npz"))
    seen_indices: set[int] = set()
    payloads: dict[str, list[np.ndarray]] = {
        "contexts": [],
        "predictions": [],
        "scores": [],
        "weights_sqrt": [],
        "weights_linear": [],
    }
    for part_path in part_paths:
        with np.load(part_path) as part:
            seen_indices.update(int(value) for value in part["completed_indices"])
            for key in payloads:
                payloads[key].append(part[key])
    if seen_indices != expected_indices:
        missing = sorted(expected_indices - seen_indices)
        unexpected = sorted(seen_indices - expected_indices)
        raise RuntimeError(
            f"{trace_set.label} incomplete: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    combined = {
        key: (
            np.concatenate(values, axis=0)
            if values
            else np.empty((0, spec.ngram), dtype=np.int32)
            if key == "contexts"
            else np.empty((0,), dtype=np.float32)
        )
        for key, values in payloads.items()
    }
    context_keys = combined["contexts"].astype(np.int64)
    pair_keys = np.concatenate(
        [
            context_keys,
            combined["predictions"].astype(np.int64)[:, None],
        ],
        axis=1,
    )
    context_indices = first_unique_indices(context_keys)
    pair_indices = first_unique_indices(pair_keys)
    all_indices = np.arange(pair_keys.shape[0], dtype=np.int64)

    scores = combined["scores"]
    sqrt_weights = combined["weights_sqrt"]
    linear_weights = combined["weights_linear"]
    base_variance = spec.mixing_alpha**2 + (1.0 - spec.mixing_alpha) ** 2
    all_positions = metric_family(
        scores,
        sqrt_weights,
        linear_weights,
        all_indices,
        base_variance,
    )
    v1 = metric_family(
        scores,
        sqrt_weights,
        linear_weights,
        context_indices,
        base_variance,
    )
    v2 = metric_family(
        scores,
        sqrt_weights,
        linear_weights,
        pair_indices,
        base_variance,
    )
    summary = {
        "label": trace_set.label,
        "trace_file": str(trace_set.path),
        "trace_count": len(expected_indices),
        "primary_scoring_method": "v1_global_context_dedup",
        "positions_before_dedup": int(pair_keys.shape[0]),
        "positions_after_global_context_dedup_v1": int(context_indices.size),
        "positions_after_global_context_prediction_dedup_v2": int(pair_indices.size),
        "duplicate_fraction_global_v1": (
            1.0 - context_indices.size / pair_keys.shape[0]
            if pair_keys.shape[0]
            else 0.0
        ),
        "duplicate_fraction_global_v2": (
            1.0 - pair_indices.size / pair_keys.shape[0]
            if pair_keys.shape[0]
            else 0.0
        ),
        "all_positions_no_dedup": all_positions,
        "global_context_dedup_v1": v1,
        "global_context_prediction_dedup_v2": v2,
        # Stable primary aliases consumed by the compact comparison summary.
        **v1,
    }
    output_path = output_dir / f"{trace_set.label}.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.checkpoint_traces < 1:
        raise ValueError("--checkpoint-traces must be positive")
    configure_runtime_caches()
    trace_sets = parse_trace_sets(args.trace_set)

    import torch

    from models.textseal import TextSealSpec

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
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
    model, tokenizer = load_model(args.model, args.adapter, args.dtype, local_rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "textseal": asdict(spec),
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "trace_sets": [asdict(trace_set) for trace_set in trace_sets],
        "world_size": world_size,
    }
    if rank == 0:
        serializable = {
            **manifest,
            "trace_sets": [
                {"label": trace_set.label, "path": str(trace_set.path)}
                for trace_set in trace_sets
            ],
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(serializable, indent=2) + "\n",
            encoding="utf-8",
        )

    expected_by_label: dict[str, set[int]] = {}
    for trace_set in trace_sets:
        rows = read_rows(trace_set.path, args.max_samples)
        expected_indices = set(range(len(rows)))
        expected_by_label[trace_set.label] = expected_indices
        work_dir = args.output_dir / ".work" / trace_set.label / f"rank_{rank:03d}"
        done = completed_indices(work_dir)
        assigned = [
            index
            for index in range(rank, len(rows), world_size)
            if index not in done
        ]
        for start in range(0, len(assigned), args.checkpoint_traces):
            indices = assigned[start : start + args.checkpoint_traces]
            arrays: list[dict[str, np.ndarray]] = []
            for batch_start in range(0, len(indices), args.batch_size):
                batch_indices = indices[batch_start : batch_start + args.batch_size]
                arrays.extend(
                    process_traces(
                        [rows[index] for index in batch_indices],
                        model=model,
                        tokenizer=tokenizer,
                        spec=spec,
                        max_seq_length=args.max_seq_length,
                        entropy_chunk=args.entropy_chunk,
                    )
                )
            part_path = work_dir / (
                f"part_{indices[0]:06d}_{indices[-1]:06d}_{len(indices):03d}.npz"
            )
            save_part(part_path, indices, arrays)
            print(
                f"rank {rank} {trace_set.label}: "
                f"{len(done) + min(start + len(indices), len(assigned))}/"
                f"{len(range(rank, len(rows), world_size))}",
                flush=True,
            )
        if world_size > 1:
            torch.distributed.barrier()

    if rank == 0:
        summaries = [
            aggregate_trace_set(
                trace_set,
                output_dir=args.output_dir,
                expected_indices=expected_by_label[trace_set.label],
                spec=spec,
            )
            for trace_set in trace_sets
        ]
        combined = {
            "model": args.model,
            "adapter": str(args.adapter) if args.adapter else None,
            "textseal": asdict(spec),
            "results": summaries,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(combined, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(combined, indent=2))

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
