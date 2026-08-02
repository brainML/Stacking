"""Deterministic fixture and frozen group-aware folds for audit tests."""

import numpy as np


def make_audit_fixture():
    rng = np.random.default_rng(20260802)
    groups = np.repeat(np.arange(6, dtype=np.int64), 2)
    latent = rng.normal(size=(12, 3))
    features = np.column_stack(
        [
            latent[:, 0],
            latent[:, 1],
            latent[:, 0] + 0.02 * rng.normal(size=12),
            np.ones(12),
        ]
    )
    targets = np.column_stack(
        [
            1.5 * latent[:, 0] - 0.7 * latent[:, 1] + 0.05 * rng.normal(size=12),
            latent[:, 2] + 0.05 * rng.normal(size=12),
        ]
    )

    # A distribution shift in the last held-out fold detects accidental
    # validation-fitted normalization.
    features[8:12, :3] += 25.0
    targets[8:12] += 10.0

    folds = (
        (np.arange(4, 12), np.arange(0, 4)),
        (np.r_[0:4, 8:12], np.arange(4, 8)),
        (np.arange(0, 8), np.arange(8, 12)),
    )
    return features, targets, groups, folds
