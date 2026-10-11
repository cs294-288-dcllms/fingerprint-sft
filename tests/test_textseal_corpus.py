from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.eval_textseal_corpus import extract_choice, read_rows, unique_pairs


def test_extract_choice_prefers_last_boxed_answer() -> None:
    assert extract_choice(r"first \boxed{A}; final \boxed{D}") == "D"
    assert extract_choice("The final answer is B.") == "B"


def test_unique_pairs_deduplicates_context_and_target() -> None:
    contexts, targets = unique_pairs([1, 2, 3, 4, 1, 2, 3, 4], ngram=3)
    keys = np.concatenate([contexts, targets[:, None]], axis=1)
    assert keys.shape[0] == np.unique(keys, axis=0).shape[0]


def test_unique_pairs_matches_official_start_position() -> None:
    contexts, targets = unique_pairs([1, 2, 3, 4, 5], ngram=3)
    assert contexts.tolist() == [[2, 3, 4]]
    assert targets.tolist() == [5]


def test_read_rows_snapshots_live_shards_by_index(tmp_path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "rank_001.jsonl").write_text(
        json.dumps({"index": 3}) + "\n" + '{"index": 5',
        encoding="utf-8",
    )
    (shards / "rank_000.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"index": 2}),
                json.dumps({"index": 1}),
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert [row["index"] for row in read_rows(shards, max_samples=2)] == [1, 2]


def test_read_rows_rejects_duplicate_shard_indices(tmp_path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    for rank in range(2):
        (shards / f"rank_{rank:03d}.jsonl").write_text(
            json.dumps({"index": 7}) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="duplicate sample indices"):
        read_rows(shards, max_samples=-1)
