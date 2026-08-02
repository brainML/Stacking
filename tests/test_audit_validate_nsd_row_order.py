import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_validate_nsd_row_order import (
    local_rows_in_trial_order,
    validate_row_order,
)


def write_metadata(path: Path) -> None:
    fieldnames = [
        "nsdId",
        "subject1",
        "subject1_rep0",
        "subject1_rep1",
        "subject1_rep2",
    ]
    rows = [
        {"nsdId": 11, "subject1": 1, "subject1_rep0": 2, "subject1_rep1": 5, "subject1_rep2": 7},
        {"nsdId": 17, "subject1": 1, "subject1_rep0": 1, "subject1_rep1": 4, "subject1_rep2": 8},
        {"nsdId": 23, "subject1": 1, "subject1_rep0": 3, "subject1_rep1": 6, "subject1_rep2": 9},
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_validate_row_order_reconstructs_metadata_order(tmp_path):
    metadata = tmp_path / "metadata.csv"
    trial_path = tmp_path / "trials.npy"
    averaged_path = tmp_path / "averaged.npy"
    write_metadata(metadata)

    trial_rows = np.asarray([1, 0, 2, 1, 0, 2, 0, 1, 2])
    image_values = np.asarray([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
    trials = image_values[trial_rows]
    np.save(trial_path, trials)
    np.save(averaged_path, image_values)

    result = validate_row_order(metadata, trial_path, averaged_path, subject=1)

    assert result["allclose"] is True
    assert result["repeat_count_min"] == 3
    assert result["repeat_count_max"] == 3
    assert result["max_absolute_error"] == 0.0


def test_duplicate_trial_position_is_rejected():
    positions = np.asarray([[1, 2, 3], [3, 4, 5]])
    with pytest.raises(ValueError, match="duplicate trial position"):
        local_rows_in_trial_order(positions, n_trials=5)


def test_validate_row_order_allows_unobserved_nominal_image(tmp_path):
    metadata = tmp_path / "metadata.csv"
    trial_path = tmp_path / "trials.npy"
    averaged_path = tmp_path / "averaged.npy"
    write_metadata(metadata)

    trial_rows = np.asarray([1, 0])
    image_values = np.asarray([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
    trials = image_values[trial_rows]
    np.save(trial_path, trials)
    np.save(averaged_path, image_values)

    result = validate_row_order(metadata, trial_path, averaged_path, subject=1)

    assert result["allclose"] is True
    assert result["images_with_no_observed_trial"] == 1
    assert result["repeat_count_min"] == 1
