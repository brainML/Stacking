import numpy as np

from scripts.audit_legacy_seed_downstream import (
    column_correlations,
    linspace_indices,
    summarize_pair,
)


def test_linspace_indices_are_fixed_and_span_half_open_chunk():
    indices = linspace_indices(0, 5000, 256)
    assert indices.shape == (256,)
    assert indices[0] == 0
    assert indices[-1] == 4999
    assert np.unique(indices).size == 256


def test_pairwise_summary_uses_prediction_and_weight_stability():
    time = np.arange(20, dtype=np.float64)[:, None]
    left_prediction = np.concatenate([time, -time], axis=1)
    right_prediction = left_prediction * 2.0
    left = {
        "stacked_prediction": left_prediction,
        "stacked_r2_raw": np.array([0.1, 0.2]),
        "weights": np.array([[0.8, 0.2], [0.1, 0.9]]),
    }
    right = {
        "stacked_prediction": right_prediction,
        "stacked_r2_raw": np.array([0.11, 0.18]),
        "weights": np.array([[0.7, 0.3], [0.2, 0.8]]),
    }
    correlations = column_correlations(left_prediction, right_prediction)
    summary = summarize_pair(left, right)
    np.testing.assert_allclose(correlations, 1.0)
    assert summary["median_voxel_prediction_correlation"] == 1.0
    assert np.isclose(summary["median_absolute_stacked_signed_r2_delta"], 0.015)
    assert np.isclose(summary["median_weight_l1_delta"], 0.2)
    assert summary["top_weight_expert_agreement"] == 1.0
