from __future__ import annotations

import numpy as np

from scripts.eval_textseal_radioactivity import (
    entropy_weights,
    first_unique_indices,
    gamma_summary,
    metric_family,
    response_positions,
)


def test_entropy_weights_are_normalized_and_concave() -> None:
    sqrt_weights, linear_weights = entropy_weights(
        np.asarray([1.0, 2.0, 5.0], dtype=np.float32)
    )
    assert np.allclose(sqrt_weights, [0.0, 0.5, 1.0])
    assert np.allclose(linear_weights, [0.1, 0.325, 1.0])


def test_gamma_summary_detects_large_scores() -> None:
    scores = np.full((1000,), 1.4, dtype=np.float32)
    weights = np.ones_like(scores)
    result = gamma_summary(scores, weights, base_variance=0.82)
    assert result["detected_at_0_01"] is True
    assert result["negative_log10_p"] > 10


def test_gamma_summary_handles_no_weight() -> None:
    result = gamma_summary(
        np.asarray([2.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        base_variance=0.82,
    )
    assert result["p_value"] == 1.0


def test_response_positions_keep_all_teacher_forced_answer_tokens() -> None:
    assert response_positions(list(range(10)), response_start=5, ngram=3) == list(
        range(5, 10)
    )


def test_first_unique_indices_support_v1_and_v2_keys() -> None:
    contexts = np.asarray([[1, 2], [1, 2], [2, 3]], dtype=np.int64)
    predictions = np.asarray([4, 5, 6], dtype=np.int64)
    assert first_unique_indices(contexts).tolist() == [0, 2]
    pairs = np.concatenate([contexts, predictions[:, None]], axis=1)
    assert first_unique_indices(pairs).tolist() == [0, 1, 2]


def test_metric_family_applies_selected_indices() -> None:
    scores = np.asarray([0.5, 1.5], dtype=np.float32)
    weights = np.ones_like(scores)
    result = metric_family(
        scores,
        weights,
        weights,
        np.asarray([1], dtype=np.int64),
        base_variance=0.82,
    )
    assert result["unweighted"]["count"] == 1
    assert result["unweighted"]["weighted_mean_score"] == 1.5
