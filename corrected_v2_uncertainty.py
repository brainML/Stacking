"""Deterministic held-out group bootstrap summaries for corrected_v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audit_reference import signed_r2


@dataclass(frozen=True)
class GroupBootstrapDifference:
    differences: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    seed: int
    resamples: int
    confidence: float


def group_bootstrap_signed_r2_difference(
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    resamples: int = 1000,
    confidence: float = 0.95,
) -> GroupBootstrapDifference:
    """Bootstrap held-out groups and report signed-R2(A)-signed-R2(B)."""

    first = np.asarray(prediction_a, dtype=np.float64)
    second = np.asarray(prediction_b, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if first.shape != second.shape or first.shape != truth.shape or first.ndim != 2:
        raise ValueError("predictions and target must be matching 2D arrays")
    if not np.isfinite(first).all() or not np.isfinite(second).all() or not np.isfinite(truth).all():
        raise ValueError("predictions and target must be finite")
    group_array = np.asarray(groups)
    if group_array.ndim != 1 or group_array.shape[0] != truth.shape[0]:
        raise ValueError("groups must contain one value per held-out row")
    unique_groups = np.unique(group_array)
    if unique_groups.size < 2:
        raise ValueError("at least two held-out groups are required")
    if resamples < 2:
        raise ValueError("resamples must be at least two")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    group_rows = [np.flatnonzero(group_array == value) for value in unique_groups]
    rng = np.random.default_rng(seed)
    differences = np.zeros((resamples, truth.shape[1]), dtype=np.float64)
    for index in range(resamples):
        draws = rng.integers(0, unique_groups.size, size=unique_groups.size)
        rows = np.concatenate([group_rows[draw] for draw in draws])
        differences[index] = signed_r2(first[rows], truth[rows]) - signed_r2(
            second[rows], truth[rows]
        )
    alpha = (1.0 - confidence) / 2.0
    mean = differences.mean(axis=0)
    lower = np.quantile(differences, alpha, axis=0)
    upper = np.quantile(differences, 1.0 - alpha, axis=0)
    for array in (differences, mean, lower, upper):
        array.setflags(write=False)
    return GroupBootstrapDifference(
        differences=differences,
        mean=mean,
        lower=lower,
        upper=upper,
        seed=int(seed),
        resamples=int(resamples),
        confidence=float(confidence),
    )
