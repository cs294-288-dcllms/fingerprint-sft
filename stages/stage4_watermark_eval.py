"""Stage 4 – student watermark evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import PeftModel
from tqdm.auto import tqdm

from config import ModelSpec, WatermarkEvalConfig
from hashing import BigramHash, HashConfig, load_hash_config
from models.loader import load_causal_lm
from utils.env import set_global_seed
from utils.io import read_jsonl_rows, write_json
from models.prompts import OASST1_SYSTEM_PROMPT, PromptBuilder
from utils.tokenization import compute_overlap, load_tokenizer

EvalEntry = Tuple[Tuple[int, int], float]
QWEN35_SCIENCE_BATCH_CAP = 12
QWEN35_SCIENCE_TOKEN_BUDGET = 4 * 4096


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _build_metric_provenance(
    cfg: WatermarkEvalConfig,
    trace_examples: int,
) -> dict[str, object]:
    """Describe the immutable inputs used to produce one Stage 4 metric."""
    return {
        "dataset": cfg.dataset,
        "trace_file": str(cfg.traces_jsonl.resolve()),
        "trace_examples": int(trace_examples),
        "trace_sha256": _sha256_file(cfg.traces_jsonl),
        "hash_config_file": str(cfg.hash_config.resolve()),
        "hash_config_sha256": _sha256_file(cfg.hash_config),
        "student_lora_dir": str(cfg.lora_dir.resolve()),
        "seed": int(cfg.seed),
    }


def _effective_batch_size(
    requested: int, dataset: str, student_model: str
) -> int:
    """Cap long-context Qwen3.5 science evaluation to a safe maximum."""
    batch_size = max(1, requested)
    if dataset.lower() == "science" and "qwen3.5" in student_model.lower():
        batch_size = min(batch_size, QWEN35_SCIENCE_BATCH_CAP)
    return batch_size


def _build_position_batches(
    positions: Sequence[int],
    token_lengths: Sequence[int],
    requested_batch_size: int,
    dataset: str,
    student_model: str,
) -> List[List[int]]:
    """Build order-preserving batches with a padding-aware token budget."""
    batch_size = _effective_batch_size(
        requested_batch_size,
        dataset,
        student_model,
    )
    if dataset.lower() != "science" or "qwen3.5" not in student_model.lower():
        return [
            list(positions[start : start + batch_size])
            for start in range(0, len(positions), batch_size)
        ]

    batches: List[List[int]] = []
    current: List[int] = []
    current_max_length = 0
    for position in positions:
        if position < 0 or position >= len(token_lengths):
            raise ValueError(f"position {position} has no token length")
        token_length = max(1, int(token_lengths[position]))
        candidate_size = len(current) + 1
        candidate_max_length = max(current_max_length, token_length)
        if current and (
            candidate_size > batch_size
            or candidate_size * candidate_max_length > QWEN35_SCIENCE_TOKEN_BUDGET
        ):
            batches.append(current)
            current = []
            current_max_length = 0
        current.append(position)
        current_max_length = max(current_max_length, token_length)
    if current:
        batches.append(current)
    return batches


def _load_stage4_checkpoint(path: Path) -> tuple[set[int], List[EvalEntry]]:
    """Load valid per-batch measurements and truncate a partial final write."""
    completed_positions: set[int] = set()
    entries: List[EvalEntry] = []
    if not path.exists():
        return completed_positions, entries
    valid_bytes = 0
    with path.open("rb") as handle:
        while line := handle.readline():
            if not line.strip():
                valid_bytes = handle.tell()
                continue
            try:
                payload = json.loads(line)
                positions = payload["positions"]
                measurements = payload["entries"]
                if (
                    not isinstance(positions, list)
                    or not all(isinstance(position, int) for position in positions)
                    or not isinstance(measurements, list)
                ):
                    break
                decoded = [
                    (
                        (int(entry["bigram"][0]), int(entry["bigram"][1])),
                        float(entry["value"]),
                    )
                    for entry in measurements
                ]
            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ):
                break
            completed_positions.update(positions)
            entries.extend(decoded)
            valid_bytes = handle.tell()
    if path.stat().st_size != valid_bytes:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
    return completed_positions, entries


def _append_stage4_checkpoint(
    path: Path,
    positions: Sequence[int],
    entries: Sequence[EvalEntry],
) -> None:
    """Durably append one completed Stage 4 batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "positions": list(positions),
        "entries": [
            {"bigram": list(bigram), "value": value}
            for bigram, value in entries
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_stage4_completion(
    completed_positions: set[int],
    expected_positions: int,
) -> None:
    """Require Stage 4 to process every local trace position."""
    expected = set(range(expected_positions))
    missing = sorted(expected - completed_positions)
    unexpected = sorted(completed_positions - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Stage 4 evaluation is incomplete or invalid: "
            f"expected_positions={expected_positions}, "
            f"completed_positions={len(completed_positions)}, "
            f"missing_positions={missing[:10]}, "
            f"unexpected_positions={unexpected[:10]}"
        )


def _filter_first_occurrences(
    bigrams: Sequence[Tuple[int, int]],
    student_positions: Sequence[int],
    sample_indices: Sequence[int],
    seen_bigrams: set[Tuple[int, int]],
) -> tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Keep only each rank's first occurrence of a bigram, preserving order."""
    if not (
        len(bigrams) == len(student_positions) == len(sample_indices)
    ):
        raise ValueError("aligned Stage 4 measurement arrays differ in length")
    kept_bigrams: List[Tuple[int, int]] = []
    kept_positions: List[int] = []
    kept_samples: List[int] = []
    for bigram, student_position, sample_index in zip(
        bigrams,
        student_positions,
        sample_indices,
    ):
        if bigram in seen_bigrams:
            continue
        seen_bigrams.add(bigram)
        kept_bigrams.append(bigram)
        kept_positions.append(int(student_position))
        kept_samples.append(int(sample_index))
    return kept_bigrams, kept_positions, kept_samples


def _aggregate_stage4_shards(
    shard_paths: Sequence[Path],
) -> tuple[int, float]:
    """Aggregate first-occurrence bigram values without a global sort."""
    seen_bigrams: set[Tuple[int, int]] = set()
    value_sum = 0.0
    value_count = 0
    for shard in shard_paths:
        _, shard_entries = _load_stage4_checkpoint(shard)
        for bigram, value in shard_entries:
            if bigram in seen_bigrams:
                continue
            seen_bigrams.add(bigram)
            value_sum += value
            value_count += 1
    mean = value_sum / value_count if value_count else 0.0
    return value_count, mean


def _build_student_shared_mask(student_tokenizer, shared_tokens: set[str]) -> torch.BoolTensor:
    """Build a mask over the student vocab for shared token strings.

    Args:
        student_tokenizer: Student tokenizer instance.
        shared_tokens: Set of token strings shared with the teacher.

    Returns:
        Boolean tensor of shape [student_vocab] marking shared tokens.
    """
    mask = torch.zeros(len(student_tokenizer), dtype=torch.bool)
    vocab = student_tokenizer.get_vocab()
    for token, idx in vocab.items():
        if token in shared_tokens:
            mask[idx] = True
    return mask


def _has_identity_token_mapping(
    source_to_target: torch.Tensor,
    shared_mask: torch.Tensor,
    teacher_vocab: int,
    student_vocab: int,
) -> bool:
    """Return whether teacher and student token ids are exactly interchangeable."""
    if teacher_vocab != student_vocab:
        return False
    if source_to_target.numel() < teacher_vocab:
        return False
    if shared_mask.numel() < student_vocab or not bool(
        shared_mask[:student_vocab].all()
    ):
        return False
    expected = torch.arange(
        teacher_vocab,
        device=source_to_target.device,
        dtype=source_to_target.dtype,
    )
    return bool(torch.equal(source_to_target[:teacher_vocab], expected))


def _aligned_offsets(
    offsets: Sequence[Tuple[int, int]],
    attention_mask: Sequence[int],
) -> Dict[int, int]:
    """Map token end offsets to actual positions in a padded batch row.

    Left padding means valid token positions are not the first ``sum(mask)``
    entries. Preserve each attended token's original tensor index so logits
    and character offsets remain aligned.
    """
    if len(offsets) != len(attention_mask):
        raise ValueError("offset and attention-mask lengths differ")
    result: Dict[int, int] = {}
    for idx, ((start, end), attended) in enumerate(zip(offsets, attention_mask)):
        if not attended or int(end) <= int(start):
            continue
        result[int(end)] = idx
    return result


def _prompt_from_row(builder: PromptBuilder, tokenizer, row: Dict, *, add_system: bool) -> str:
    """Extract or build a prompt from a trace row.

    Args:
        builder: PromptBuilder used for message-based prompts.
        tokenizer: Tokenizer used in the prompt builder.
        row: Trace row dict with "prompt" or "messages".
        add_system: Whether to insert a system prompt for messages.

    Returns:
        Prompt text ready for concatenation with the response.
    """
    if "messages" in row:
        messages = row.get("messages") or []
        return builder.build_from_messages(tokenizer, messages, add_system=add_system)
    prompt = row.get("prompt")
    if not prompt:
        raise ValueError("Trace row missing prompt/messages")
    return prompt


def run_stage4(cfg: WatermarkEvalConfig) -> Path:
    """Run Stage 4 to evaluate watermark statistics for a student model.

    Args:
        cfg: WatermarkEvalConfig with dataset, models, and evaluation settings.

    Returns:
        Path to the written watermark JSON file.
    """
    accelerator = Accelerator()
    set_global_seed(cfg.seed)

    traces = read_jsonl_rows(cfg.traces_jsonl)
    if not traces:
        raise RuntimeError("Trace file is empty")

    hash_cfg = load_hash_config(cfg.hash_config)
    teacher_tokenizer = load_tokenizer(cfg.teacher, padding_side="left")
    student_tokenizer = load_tokenizer(cfg.student, padding_side="left")
    add_system_for_messages = cfg.dataset == "oasst1"
    builder = PromptBuilder(system_prompt=OASST1_SYSTEM_PROMPT if add_system_for_messages else None)

    hash_fn = BigramHash(
        hash_cfg,
        vocab_size=len(teacher_tokenizer),
        excluded_token_ids=getattr(teacher_tokenizer, "all_special_ids", None),
    )

    overlap = compute_overlap(teacher_tokenizer, student_tokenizer)
    shared_mask = _build_student_shared_mask(student_tokenizer, overlap.shared_token_strings)
    student_vocab = len(student_tokenizer)
    map_tensor = overlap.source_to_target
    teacher_vocab = len(teacher_tokenizer)
    identity_token_mapping = _has_identity_token_mapping(
        map_tensor,
        shared_mask,
        teacher_vocab,
        student_vocab,
    )
    map_dev = map_tensor.to(accelerator.device)
    if map_dev.shape[0] < teacher_vocab:
        padded = torch.full((teacher_vocab,), -1, device=accelerator.device, dtype=torch.long)
        padded[: map_dev.shape[0]] = map_dev
        map_dev = padded
    else:
        map_dev = map_dev[:teacher_vocab]
    sentinel = student_vocab
    map_indices = map_dev.clone()
    map_indices[map_indices < 0] = sentinel

    base_model = load_causal_lm(cfg.student)
    base_model.resize_token_embeddings(len(student_tokenizer))
    student_model = PeftModel.from_pretrained(base_model, cfg.lora_dir)
    if hasattr(student_model, "config"):
        student_model.config.use_cache = False
    student_model.to(accelerator.device)
    student_model.eval()

    local_rows = traces[accelerator.process_index :: accelerator.num_processes]
    local_texts: List[str] = []
    local_prompt_lengths: List[int] = []
    local_token_lengths: List[int] = []
    for row in local_rows:
        prompt_text = _prompt_from_row(
            builder,
            student_tokenizer,
            row,
            add_system=add_system_for_messages,
        )
        text = prompt_text + (row.get("response") or "")
        local_texts.append(text)
        local_prompt_lengths.append(len(prompt_text))
        local_token_lengths.append(
            len(
                student_tokenizer(
                    text,
                    add_special_tokens=False,
                )["input_ids"]
            )
        )
    tmp_dir = cfg.output_path.parent / f"_tmp_stage4_{cfg.output_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rank_path = tmp_dir / f"rank_{accelerator.process_index:03d}.jsonl"

    student_shared_mask = shared_mask
    completed_positions, completed_entries = _load_stage4_checkpoint(rank_path)
    seen_bigrams = {bigram for bigram, _ in completed_entries}

    batch_size = _effective_batch_size(
        cfg.batch_size,
        cfg.dataset,
        cfg.student.name,
    )
    if accelerator.is_local_main_process and batch_size != max(1, cfg.batch_size):
        print(
            "Stage 4 memory guard: "
            f"batch {cfg.batch_size} -> {batch_size} for "
            f"{cfg.dataset}/{cfg.student.name}.",
            flush=True,
        )
    pending_positions = [
        position
        for position in range(len(local_rows))
        if position not in completed_positions
    ]
    position_batches = _build_position_batches(
        pending_positions,
        local_token_lengths,
        batch_size,
        cfg.dataset,
        cfg.student.name,
    )
    iterator = position_batches
    if accelerator.is_local_main_process:
        iterator = tqdm(
            iterator,
            total=len(position_batches),
            desc="Stage 4: watermark eval",
        )

    for positions in iterator:
        batch = [local_rows[position] for position in positions]
        batch_values: List[EvalEntry] = []
        texts = [local_texts[position] for position in positions]
        prompt_lengths = [local_prompt_lengths[position] for position in positions]
        # Tokenize with student tokenizer to get offsets/token ids.
        student_inputs = student_tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        attention_mask = student_inputs["attention_mask"].to(accelerator.device)
        input_ids = student_inputs["input_ids"].to(accelerator.device)
        offsets_batch = student_inputs["offset_mapping"]
        with torch.no_grad():
            logits = student_model(input_ids=input_ids, attention_mask=attention_mask).logits

        batch_bigrams: List[Tuple[int, int]] = []
        batch_student_positions: List[int] = []
        batch_sample_indices: List[int] = []

        for idx, row in enumerate(batch):
            text = texts[idx]
            # Teacher tokenization to find bigrams + offsets on teacher vocab.
            teacher_encoding = teacher_tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            teacher_ids = torch.tensor(teacher_encoding["input_ids"], dtype=torch.long)
            teacher_offsets = [
                (int(start), int(end)) for start, end in teacher_encoding["offset_mapping"]
            ]
            offsets_list = [
                (int(start), int(end))
                for start, end in offsets_batch[idx]
            ]
            alignment = _aligned_offsets(
                offsets_list, attention_mask[idx].tolist()
            )
            response_start = prompt_lengths[idx]
            for t_idx, (_, end) in enumerate(teacher_offsets):
                if end <= response_start:
                    continue
                if end not in alignment:
                    continue
                if t_idx < 1:
                    continue
                s_idx = alignment[end]
                if s_idx < 0 or s_idx >= input_ids.shape[1]:
                    continue
                bigram = (int(teacher_ids[t_idx - 1]), int(teacher_ids[t_idx]))
                batch_bigrams.append(bigram)
                batch_student_positions.append(int(s_idx))
                batch_sample_indices.append(int(idx))

        (
            batch_bigrams,
            batch_student_positions,
            batch_sample_indices,
        ) = _filter_first_occurrences(
            batch_bigrams,
            batch_student_positions,
            batch_sample_indices,
            seen_bigrams,
        )
        if not batch_bigrams:
            _append_stage4_checkpoint(rank_path, positions, batch_values)
            continue

        mask_chunk = max(1, cfg.mask_chunk)
        shared = student_shared_mask.to(accelerator.device)
        for offset in range(0, len(batch_bigrams), mask_chunk):
            bg_chunk = batch_bigrams[offset : offset + mask_chunk]
            sample_chunk = batch_sample_indices[offset : offset + mask_chunk]
            pos_chunk = batch_student_positions[offset : offset + mask_chunk]

            sample_idx_tensor = torch.tensor(sample_chunk, device=accelerator.device, dtype=torch.long)
            token_idx_tensor = torch.tensor(pos_chunk, device=accelerator.device, dtype=torch.long)
            selected_logits = logits[sample_idx_tensor, token_idx_tensor]

            probs = F.softmax(selected_logits, dim=-1)
            if cfg.mode == "closed" and identity_token_mapping:
                samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
                shared_flags = shared[samples]
                hits = hash_fn.membership_batch(
                    bg_chunk,
                    samples,
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                for bigram, is_shared, hit in zip(
                    bg_chunk,
                    shared_flags.tolist(),
                    hits.tolist(),
                ):
                    if not is_shared:
                        continue
                    batch_values.append((bigram, float(hit)))
                continue

            teacher_masks = hash_fn.mask_batch(
                bg_chunk,
                device=accelerator.device,
                dtype=torch.bool,
            )
            if identity_token_mapping:
                student_masks = teacher_masks
            else:
                teacher_mask_vocab = teacher_masks.shape[1]
                map_slice = map_indices[:teacher_mask_vocab]
                student_masks = torch.zeros(
                    (teacher_masks.shape[0], student_vocab + 1),
                    device=accelerator.device,
                    dtype=torch.bool,
                )
                student_masks.scatter_(
                    1,
                    map_slice.unsqueeze(0).expand_as(teacher_masks),
                    teacher_masks,
                )
                student_masks = student_masks[:, :student_vocab]

            if cfg.mode == "open":
                shared_mass = (probs * shared).sum(dim=-1)
                green_mass = (probs * student_masks).sum(dim=-1)
                values = (green_mass / shared_mass).tolist()
                for bigram, value in zip(bg_chunk, values):
                    batch_values.append((bigram, float(value)))
            else:
                samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
                shared_flags = shared[samples]
                hits = student_masks.gather(1, samples.unsqueeze(-1)).squeeze(-1).to(torch.float32)
                for bigram, is_shared, hit in zip(bg_chunk, shared_flags.tolist(), hits.tolist()):
                    if not is_shared:
                        continue  # discard samples outside the shared vocabulary
                    batch_values.append((bigram, float(hit)))

        _append_stage4_checkpoint(rank_path, positions, batch_values)

    completed_positions.update(pending_positions)
    _validate_stage4_completion(
        completed_positions,
        expected_positions=len(local_rows),
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        shards = [
            tmp_dir / f"rank_{idx:03d}.jsonl"
            for idx in range(accelerator.num_processes)
            if (tmp_dir / f"rank_{idx:03d}.jsonl").exists()
        ]
        num_measurements, mean = _aggregate_stage4_shards(shards)
        if not num_measurements:
            payload = {
                "num_measurements": 0,
                "mean": 0.0,
                "mode": cfg.mode,
                "supervision": cfg.supervision,
                "note": "no_measurements_collected",
            }
        else:
            payload = {
                "num_measurements": num_measurements,
                "mean": mean,
                "mode": cfg.mode,
                "supervision": cfg.supervision,
            }
        payload.update(_build_metric_provenance(cfg, len(traces)))
        write_json(cfg.output_path, payload)
        for shard in tmp_dir.glob("rank_*.jsonl"):
            shard.unlink()
        tmp_dir.rmdir()
    accelerator.wait_for_everyone()
    return cfg.output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for Stage 4.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Stage 4 – watermark evaluation")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--hash-config", type=Path, required=True)
    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--teacher-dtype", type=str, default="bfloat16")
    parser.add_argument("--teacher-pad-token", type=str, default=None)
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--student-dtype", type=str, default="bfloat16")
    parser.add_argument("--student-pad-token", type=str, default=None)
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["open", "closed"], default="open")
    parser.add_argument("--supervision", choices=["supervised", "unsupervised"], default="supervised")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mask-chunk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for Stage 4.

    Args:
        argv: Optional list of CLI arguments (defaults to sys.argv).

    Returns:
        None.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = WatermarkEvalConfig(
        dataset=args.dataset,
        teacher=ModelSpec(name=args.teacher_model, dtype=args.teacher_dtype, pad_token=args.teacher_pad_token),
        student=ModelSpec(name=args.student_model, dtype=args.student_dtype, pad_token=args.student_pad_token),
        hash_config=args.hash_config,
        traces_jsonl=args.traces,
        lora_dir=args.lora_dir,
        mode=args.mode,  # type: ignore[arg-type]
        supervision=args.supervision,  # type: ignore[arg-type]
        output_path=args.output,
        batch_size=args.batch_size,
        seed=args.seed,
        mask_chunk=args.mask_chunk,
    )
    run_stage4(cfg)


if __name__ == "__main__":
    main()
