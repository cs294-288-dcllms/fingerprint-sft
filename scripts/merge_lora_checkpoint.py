#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter into a local Hugging Face checkpoint."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    complete_file = args.output / "merge_manifest.json"
    if complete_file.exists():
        manifest = json.loads(complete_file.read_text(encoding="utf-8"))
        if (
            manifest.get("base_model") == args.base_model
            and manifest.get("adapter") == str(args.adapter.resolve())
        ):
            print(f"Merged checkpoint already complete: {args.output}")
            return

    adapter_config = args.adapter / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"Missing LoRA adapter config: {adapter_config}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
            local_files_only=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            str(args.adapter),
            local_files_only=True,
        )
        model = model.merge_and_unload(safe_merge=True)
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(temporary)
        manifest = {
            "base_model": args.base_model,
            "adapter": str(args.adapter.resolve()),
            "dtype": "bfloat16",
        }
        (temporary / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.output.exists():
            shutil.rmtree(args.output)
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    print(f"Merged checkpoint written to {args.output}")


if __name__ == "__main__":
    main()
