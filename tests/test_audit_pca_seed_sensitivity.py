from pathlib import Path

import numpy as np

from scripts.audit_pca_seed_sensitivity import run_seed_sensitivity


def test_seed_sensitivity_reports_pairwise_invariants(tmp_path: Path):
    rng = np.random.RandomState(7)
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, rng.standard_normal((80, 30)).astype(np.float32))

    result = run_seed_sensitivity(
        feature_path,
        seeds=[0, 1, 1337],
        n_components=5,
        svd_solver="randomized",
        probe_row_count=12,
    )

    assert result["status"] == "completed"
    assert result["response_data_used"] is False
    assert len(result["runs"]) == 3
    assert len(result["pairwise"]) == 3
    assert all(item["resolved_svd_solver"] == "randomized" for item in result["runs"])
    for comparison in result["pairwise"]:
        assert 0 <= comparison["principal_cosine_min"] <= 1.0000001
        assert comparison["subspace_chordal_distance"] >= 0
        assert comparison["probe_gram_relative_frobenius_difference"] >= 0
