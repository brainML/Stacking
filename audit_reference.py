"""Small correctness utilities for the Stacking foundation audit.

This module is not yet the complete corrected_v2 estimator. It freezes the
train-fitted preprocessing and externally supplied split behavior needed by
the first audit packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


IndexSplit = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Standardizer:
    """Column standardizer fitted on one declared training partition."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2:
            raise ValueError("values must be a finite 2D array with at least two rows")
        if not np.isfinite(array).all():
            raise ValueError("values contain non-finite entries")
        mean = array.mean(axis=0)
        scale = array.std(axis=0)
        scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
        mean.setflags(write=False)
        scale.setflags(write=False)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.mean.shape[0]:
            raise ValueError("values do not match the fitted column identity")
        if not np.isfinite(array).all():
            raise ValueError("values contain non-finite entries")
        return (array - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.mean.shape[0]:
            raise ValueError("values do not match the fitted column identity")
        if not np.isfinite(array).all():
            raise ValueError("values contain non-finite entries")
        return array * self.scale + self.mean


def _immutable_indices(values: Iterable[int], *, n_rows: int, name: str) -> np.ndarray:
    indices = np.asarray(tuple(values), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{name} must be a nonempty 1D index array")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} contains duplicate rows")
    if np.any(indices < 0) or np.any(indices >= n_rows):
        raise ValueError(f"{name} contains out-of-range rows")
    indices.setflags(write=False)
    return indices


def validate_splits(
    n_rows: int,
    splits: Sequence[Tuple[Iterable[int], Iterable[int]]],
    *,
    groups: Optional[np.ndarray] = None,
    require_partition: bool = True,
) -> Tuple[IndexSplit, ...]:
    """Validate and freeze externally supplied cross-validation splits."""

    if n_rows < 2:
        raise ValueError("n_rows must be at least two")
    if not splits:
        raise ValueError("at least one split is required")
    group_array = None
    if groups is not None:
        group_array = np.asarray(groups)
        if group_array.ndim != 1 or group_array.shape[0] != n_rows:
            raise ValueError("groups must contain one value per row")

    frozen = []
    validation_counts = np.zeros(n_rows, dtype=np.int64)
    all_rows = np.arange(n_rows)
    for fold, (fit_values, validation_values) in enumerate(splits):
        fit = _immutable_indices(fit_values, n_rows=n_rows, name=f"fit[{fold}]")
        validation = _immutable_indices(
            validation_values, n_rows=n_rows, name=f"validation[{fold}]"
        )
        if np.intersect1d(fit, validation).size:
            raise ValueError(f"split {fold} has row leakage")
        if require_partition:
            expected_fit = np.setdiff1d(all_rows, validation, assume_unique=True)
            if not np.array_equal(np.sort(fit), expected_fit):
                raise ValueError(f"split {fold} does not partition all rows")
        if group_array is not None:
            fit_groups = np.unique(group_array[fit])
            validation_groups = np.unique(group_array[validation])
            if np.intersect1d(fit_groups, validation_groups).size:
                raise ValueError(f"split {fold} has group leakage")
        validation_counts[validation] += 1
        frozen.append((fit, validation))

    if require_partition and not np.all(validation_counts == 1):
        raise ValueError("validation rows must form an exact one-pass partition")
    return tuple(frozen)


def signed_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-column held-out R2 with constant targets defined as zero."""

    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape or prediction_array.ndim != 2:
        raise ValueError("prediction and target must be matching 2D arrays")
    if not np.isfinite(prediction_array).all() or not np.isfinite(target_array).all():
        raise ValueError("prediction and target must be finite")
    residual = np.mean((target_array - prediction_array) ** 2, axis=0)
    total = np.var(target_array, axis=0)
    return np.divide(
        total - residual,
        total,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > np.finfo(np.float64).eps,
    )
