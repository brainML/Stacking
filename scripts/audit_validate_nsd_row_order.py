#!/usr/bin/env python3
"""Validate averaged NSD response rows against trial-order responses.

This is a read-only audit utility. It reconstructs per-image means for a
deterministic sample of cortical columns using repeat positions in the NSD
stimulus metadata. It emits only compact identities and numerical diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical_int64_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def load_subject_order(metadata_path: Path, subject: int) -> tuple[np.ndarray, np.ndarray]:
    membership_key = f"subject{subject}"
    repeat_keys = [f"subject{subject}_rep{i}" for i in range(3)]
    nsd_ids: list[int] = []
    repeat_positions: list[list[int]] = []

    with metadata_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"nsdId", membership_key, *repeat_keys}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"metadata is missing columns: {sorted(missing)}")
        for row in reader:
            if int(row[membership_key]) != 1:
                continue
            nsd_ids.append(int(row["nsdId"]))
            repeat_positions.append([int(row[key]) for key in repeat_keys])

    ids = np.asarray(nsd_ids, dtype=np.int64)
    positions = np.asarray(repeat_positions, dtype=np.int64)
    if ids.ndim != 1 or positions.shape != (ids.size, 3):
        raise ValueError("invalid subject metadata shape")
    if np.unique(ids).size != ids.size:
        raise ValueError("subject NSD IDs are not unique")
    return ids, positions


def local_rows_in_trial_order(repeat_positions: np.ndarray, n_trials: int) -> np.ndarray:
    trial_rows = np.full(n_trials, -1, dtype=np.int64)
    for local_row, positions in enumerate(repeat_positions):
        for one_based_position in positions:
            if one_based_position <= 0 or one_based_position > n_trials:
                continue
            index = one_based_position - 1
            if trial_rows[index] != -1:
                raise ValueError(f"duplicate trial position {one_based_position}")
            trial_rows[index] = local_row
    missing = np.flatnonzero(trial_rows < 0)
    if missing.size:
        raise ValueError(f"metadata leaves {missing.size} trial positions unmapped")
    return trial_rows


def validate_row_order(
    metadata_path: Path,
    trial_path: Path,
    averaged_path: Path,
    subject: int,
    sample_images: int = 32,
    sample_columns: int = 16,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> dict[str, object]:
    ids, repeat_positions = load_subject_order(metadata_path, subject)
    trials = np.load(trial_path, mmap_mode="r")
    averaged = np.load(averaged_path, mmap_mode="r")
    if trials.ndim != 2 or averaged.ndim != 2:
        raise ValueError("trial and averaged responses must be two-dimensional")
    if averaged.shape[0] != ids.size:
        raise ValueError("averaged response rows do not match subject image count")
    if trials.shape[1] != averaged.shape[1]:
        raise ValueError("trial and averaged responses have different voxel counts")
    if sample_images < 1 or sample_columns < 1:
        raise ValueError("sample_images and sample_columns must be positive")

    trial_rows = local_rows_in_trial_order(repeat_positions, trials.shape[0])
    column_count = min(sample_columns, trials.shape[1])
    columns = np.unique(
        np.linspace(0, trials.shape[1] - 1, column_count, dtype=np.int64)
    )
    counts = np.bincount(trial_rows, minlength=ids.size)
    observed_image_rows = np.flatnonzero(counts > 0)
    unobserved_image_rows = np.flatnonzero(counts == 0)
    if not observed_image_rows.size:
        raise ValueError("no subject image has an observed trial")
    present_repeat_counts = np.unique(counts[observed_image_rows])
    per_group = max(1, int(np.ceil(sample_images / present_repeat_counts.size)))
    image_row_groups = []
    for repeat_count in present_repeat_counts:
        candidates = np.flatnonzero(counts == repeat_count)
        group_size = min(per_group, candidates.size)
        selected = np.unique(
            np.linspace(0, candidates.size - 1, group_size, dtype=np.int64)
        )
        image_row_groups.append(candidates[selected])
    image_rows = np.sort(np.concatenate(image_row_groups))
    selected_trial_indices = np.flatnonzero(np.isin(trial_rows, image_rows))
    selected_local_rows = trial_rows[selected_trial_indices]
    row_lookup = np.full(ids.size, -1, dtype=np.int64)
    row_lookup[image_rows] = np.arange(image_rows.size)
    compact_rows = row_lookup[selected_local_rows]
    trial_sample = np.asarray(
        trials[np.ix_(selected_trial_indices, columns)], dtype=np.float64
    )
    sums = np.zeros((image_rows.size, columns.size), dtype=np.float64)
    np.add.at(sums, compact_rows, trial_sample)
    reconstructed = sums / counts[image_rows, None]
    padded_three_repeat_mean = sums / 3.0
    observed = np.asarray(averaged[np.ix_(image_rows, columns)], dtype=np.float64)
    difference = reconstructed - observed
    padded_difference = padded_three_repeat_mean - observed
    finite_difference = np.abs(difference[np.isfinite(difference)])
    finite_padded_difference = np.abs(padded_difference[np.isfinite(padded_difference)])
    unobserved_sample_size = min(per_group, unobserved_image_rows.size)
    if unobserved_sample_size:
        unobserved_selected = np.unique(
            np.linspace(
                0,
                unobserved_image_rows.size - 1,
                unobserved_sample_size,
                dtype=np.int64,
            )
        )
        sampled_unobserved_rows = unobserved_image_rows[unobserved_selected]
        unobserved_values = np.asarray(
            averaged[np.ix_(sampled_unobserved_rows, columns)], dtype=np.float64
        )
        finite_unobserved = np.abs(unobserved_values[np.isfinite(unobserved_values)])
        unobserved_rows_all_zero = bool(
            np.allclose(unobserved_values, 0.0, rtol=0.0, atol=atol, equal_nan=False)
        )
        unobserved_rows_max_absolute_value = float(
            finite_unobserved.max(initial=0.0)
        )
    else:
        sampled_unobserved_rows = np.asarray([], dtype=np.int64)
        unobserved_rows_all_zero = None
        unobserved_rows_max_absolute_value = None
    errors_by_repeat_count: dict[str, dict[str, float | int | bool]] = {}
    for repeat_count in present_repeat_counts:
        group = counts[image_rows] == repeat_count
        group_difference = difference[group]
        group_padded_difference = padded_difference[group]
        group_finite = np.abs(group_difference[np.isfinite(group_difference)])
        group_padded_finite = np.abs(
            group_padded_difference[np.isfinite(group_padded_difference)]
        )
        errors_by_repeat_count[str(int(repeat_count))] = {
            "sampled_images": int(np.count_nonzero(group)),
            "observed_mean_allclose": bool(
                np.allclose(
                    reconstructed[group], observed[group], rtol=rtol, atol=atol, equal_nan=True
                )
            ),
            "observed_mean_max_absolute_error": float(group_finite.max(initial=0.0)),
            "padded_three_repeat_allclose": bool(
                np.allclose(
                    padded_three_repeat_mean[group],
                    observed[group],
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
            ),
            "padded_three_repeat_max_absolute_error": float(
                group_padded_finite.max(initial=0.0)
            ),
        }

    return {
        "subject": subject,
        "nsd_images": int(ids.size),
        "trials": int(trials.shape[0]),
        "voxels": int(trials.shape[1]),
        "sampled_image_rows": image_rows.tolist(),
        "sampled_columns": columns.tolist(),
        "images_with_no_observed_trial": int(np.count_nonzero(counts == 0)),
        "sampled_unobserved_image_rows": sampled_unobserved_rows.tolist(),
        "sampled_unobserved_rows_all_zero": unobserved_rows_all_zero,
        "sampled_unobserved_rows_max_absolute_value": unobserved_rows_max_absolute_value,
        "repeat_count_min": int(counts[observed_image_rows].min()),
        "repeat_count_max": int(counts.max()),
        "image_order_nsd_id_int64_sha256": canonical_int64_sha256(ids),
        "trial_order_nsd_id_int64_sha256": canonical_int64_sha256(ids[trial_rows]),
        "allclose": bool(np.allclose(reconstructed, observed, rtol=rtol, atol=atol, equal_nan=True)),
        "max_absolute_error": float(finite_difference.max(initial=0.0)),
        "mean_absolute_error": float(finite_difference.mean()) if finite_difference.size else 0.0,
        "padded_three_repeat_allclose": bool(
            np.allclose(
                padded_three_repeat_mean, observed, rtol=rtol, atol=atol, equal_nan=True
            )
        ),
        "padded_three_repeat_max_absolute_error": float(
            finite_padded_difference.max(initial=0.0)
        ),
        "errors_by_repeat_count": errors_by_repeat_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--averaged", type=Path, required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--sample-images", type=int, default=32)
    parser.add_argument("--sample-columns", type=int, default=16)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_row_order(
        metadata_path=args.metadata,
        trial_path=args.trials,
        averaged_path=args.averaged,
        subject=args.subject,
        sample_images=args.sample_images,
        sample_columns=args.sample_columns,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
