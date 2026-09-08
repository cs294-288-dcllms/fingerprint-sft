#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)
ANSWER_PATTERN = re.compile(
    r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*[*\\(\[]*([A-D])\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base/LoRA models on the Science MCQ split."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2_048)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16")
    return parser.parse_args()


def extract_choice(text: str) -> str | None:
    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        return boxed[-1].upper()
    answers = ANSWER_PATTERN.findall(text)
    return answers[-1].upper() if answers else None


def is_correct(prediction: str, gold: str) -> bool:
    predicted_choice = extract_choice(prediction)
    gold_choice = extract_choice(gold)
    return (
        predicted_choice is not None and gold_choice is not None and predicted_choice == gold_choice
    )


def read_dataset(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_number} has invalid messages")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{line_number} does not end with an assistant message")
            rows.append(row)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    prompt_messages = messages[:-1]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(prompt_messages, enable_thinking=True, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(prompt_messages, **kwargs)


def load_model(
    model_name: str,
    adapter: Path | None,
    dtype_name: str,
    local_rank: int | None,
) -> tuple[Any, Any]:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    load_kwargs = {
        "config": config,
        "torch_dtype": dtype,
        "device_map": {"": local_rank} if local_rank is not None else "auto",
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        **load_kwargs,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(adapter),
            local_files_only=True,
        )
    model.eval()
    return model, tokenizer


def _load_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    """Load valid checkpoint rows and truncate any malformed tail."""
    if not path.exists():
        return []
    checkpoint_rows: list[dict[str, Any]] = []
    valid_bytes = 0
    with path.open("rb") as handle:
        while line := handle.readline():
            if not line.strip():
                valid_bytes = handle.tell()
                continue
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                break
            if not isinstance(payload, dict) or "index" not in payload:
                break
            checkpoint_rows.append(payload)
            valid_bytes = handle.tell()
    if path.stat().st_size != valid_bytes:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
    return checkpoint_rows


def _validate_complete_predictions(
    rows: list[dict[str, Any]], expected: int
) -> list[dict[str, Any]]:
    """Require exactly one prediction for every expected dataset index."""
    seen: set[int] = set()
    duplicates: set[int] = set()
    non_integer_count = 0
    for row in rows:
        index = row.get("index")
        if not isinstance(index, int):
            non_integer_count += 1
            continue
        if index in seen:
            duplicates.add(index)
        seen.add(index)
    expected_indices = set(range(expected))
    missing = sorted(expected_indices - seen)
    unexpected = sorted(seen - expected_indices)
    if (
        len(rows) != expected
        or non_integer_count
        or duplicates
        or missing
        or unexpected
    ):
        raise RuntimeError(
            "Science evaluation merge is incomplete or invalid: "
            f"expected_rows={expected}, actual_rows={len(rows)}, "
            f"non_integer_indices={non_integer_count}, "
            f"missing_indices={missing[:10]}, "
            f"unexpected_indices={unexpected[:10]}, "
            f"duplicate_indices={sorted(duplicates)[:10]}"
        )
    return sorted(rows, key=lambda row: row["index"])


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install the pinned SFT environment before evaluation.") from exc

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if world_size > 1 else None
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )

    rows = read_dataset(args.dataset, args.max_samples)
    model, tokenizer = load_model(args.model, args.adapter, args.dtype, local_rank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shard_output = (
        args.output.with_suffix(args.output.suffix + f".rank{rank}.tmp")
        if world_size > 1
        else args.output.with_suffix(args.output.suffix + ".tmp")
    )
    all_assigned_indices = list(range(rank, len(rows), world_size))
    existing_rows = _load_checkpoint_rows(shard_output)
    completed_indices = {int(item["index"]) for item in existing_rows}
    assigned_indices = [index for index in all_assigned_indices if index not in completed_indices]
    correct = sum(bool(item.get("correct")) for item in existing_rows)
    parsed = sum(item.get("predicted_choice") is not None for item in existing_rows)
    completed_before = len(existing_rows)

    with shard_output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(assigned_indices), args.batch_size):
            batch_indices = assigned_indices[start : start + args.batch_size]
            batch_rows = [rows[index] for index in batch_indices]
            prompts = [render_prompt(tokenizer, row["messages"]) for row in batch_rows]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            input_device = next(model.parameters()).device
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = outputs[:, inputs["input_ids"].shape[1] :]
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for index, row, prediction in zip(batch_indices, batch_rows, predictions, strict=True):
                gold_text = str(row["messages"][-1]["content"])
                predicted_choice = extract_choice(prediction)
                gold_choice = extract_choice(gold_text)
                item_correct = is_correct(prediction, gold_text)
                parsed += int(predicted_choice is not None)
                correct += int(item_correct)
                payload = {
                    "index": index,
                    "prompt_sha256": row.get("_meta", {}).get("prompt_sha256"),
                    "question": row["messages"][-2]["content"],
                    "gold": gold_text,
                    "gold_choice": gold_choice,
                    "prediction": prediction,
                    "predicted_choice": predicted_choice,
                    "correct": item_correct,
                    "model": args.model,
                    "adapter": str(args.adapter) if args.adapter else None,
                }
                json.dump(payload, handle, ensure_ascii=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed = completed_before + min(start + args.batch_size, len(assigned_indices))
            print(
                f"rank {rank}: evaluated {completed}/{len(all_assigned_indices)}",
                flush=True,
            )

    if world_size > 1:
        counts = torch.tensor([correct, parsed], device=f"cuda:{local_rank}", dtype=torch.long)
        torch.distributed.all_reduce(counts)
        correct, parsed = (int(value) for value in counts.tolist())
        torch.distributed.barrier()
        if rank == 0:
            merged = []
            for shard_rank in range(world_size):
                shard_path = args.output.with_suffix(args.output.suffix + f".rank{shard_rank}.tmp")
                with shard_path.open("r", encoding="utf-8") as handle:
                    merged.extend(json.loads(line) for line in handle if line.strip())
            merged = _validate_complete_predictions(merged, len(rows))
            with args.output.open("w", encoding="utf-8") as handle:
                for row in merged:
                    json.dump(row, handle, ensure_ascii=False)
                    handle.write("\n")
            for shard_rank in range(world_size):
                args.output.with_suffix(args.output.suffix + f".rank{shard_rank}.tmp").unlink(
                    missing_ok=True
                )
        torch.distributed.barrier()
    else:
        _validate_complete_predictions(_load_checkpoint_rows(shard_output), len(rows))
        shard_output.replace(args.output)

    if rank == 0:
        sample_count = len(rows)
        summary = {
            "model": args.model,
            "adapter": str(args.adapter) if args.adapter else None,
            "dataset": str(args.dataset),
            "samples": sample_count,
            "parsed": parsed,
            "parse_rate": parsed / sample_count if sample_count else 0.0,
            "correct": correct,
            "accuracy": correct / sample_count if sample_count else 0.0,
            "predictions": str(args.output),
        }
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
