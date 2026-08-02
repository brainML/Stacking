#!/usr/bin/env python3
"""Finalize a downstream seed audit from completed private seed artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.audit_legacy_seed_downstream import (
        _atomic_json,
        _gate_passes,
        array_sha256,
        linspace_indices,
        sha256_file,
        summarize_pair,
    )
except ModuleNotFoundError:  # flat deployment directory on the compute host
    from audit_legacy_seed_downstream import (  # type: ignore[no-redef]
        _atomic_json,
        _gate_passes,
        array_sha256,
        linspace_indices,
        sha256_file,
        summarize_pair,
    )


REQUIRED_ARRAYS = {
    "voxel_indices",
    "test_rows",
    "pca_transform_hashes_ascii",
    "chosen_lambdas",
    "weights",
    "expert_r2_raw",
    "expert_r2_z",
    "train_r2",
    "stacked_train_r2",
    "stacked_r2_raw",
    "stacked_r2_z",
    "stacked_prediction",
}


def load_seed_artifact(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing seed artifact: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            raise ValueError(f"{path.name} missing arrays: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def finalize(config: dict[str, Any], output_directory: Path) -> dict[str, Any]:
    if not output_directory.is_dir():
        raise FileNotFoundError("existing output directory is required")
    summary_path = output_directory / "summary.json"
    if summary_path.exists():
        raise FileExistsError("refusing to overwrite existing summary")

    selection = config["voxel_selection"]
    expected_voxels = linspace_indices(
        selection["source_chunk_start"],
        selection["source_chunk_stop"],
        selection["count"],
    )
    seeds = [int(seed) for seed in config["pca"]["initial_numpy_seeds"]]
    seed_results: dict[int, dict[str, np.ndarray]] = {}
    seed_runs = []
    for seed in seeds:
        artifact_path = output_directory / f"seed_{seed}.npz"
        arrays = load_seed_artifact(artifact_path)
        if not np.array_equal(arrays["voxel_indices"], expected_voxels):
            raise ValueError(f"voxel-index mismatch in {artifact_path.name}")
        pca_hashes = [
            value.decode("ascii")
            for value in arrays["pca_transform_hashes_ascii"].tolist()
        ]
        if len(pca_hashes) != len(config["feature_order"]):
            raise ValueError(f"PCA hash-count mismatch in {artifact_path.name}")
        seed_results[seed] = arrays
        seed_runs.append(
            {
                "seed": seed,
                "elapsed_seconds": None,
                "elapsed_recovery_note": "not persisted before summary-only failure",
                "train_row_count": config["outer_evaluation"]["train_rows"],
                "test_row_count": int(arrays["test_rows"].size),
                "voxel_count": int(arrays["voxel_indices"].size),
                "pca_transform_hashes": pca_hashes,
                "private_artifact_sha256": sha256_file(artifact_path),
            }
        )

    predictive = config["decision_gates"]["predictive_stability_all_pairs"]
    interpretive = config["decision_gates"]["interpretive_instability_flags"]
    pairs = []
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            metrics = summarize_pair(seed_results[left_seed], seed_results[right_seed])
            metrics.update(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "predictive_stability_pass": _gate_passes(metrics, predictive),
                    "interpretive_instability_flag": bool(
                        metrics["top_weight_expert_agreement"]
                        < interpretive["top_weight_expert_agreement_min"]
                        or metrics["median_weight_l1_delta"]
                        > interpretive["median_weight_l1_delta_max"]
                    ),
                }
            )
            pairs.append(metrics)

    result = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": "completed_after_summary_recovery",
        "response_data_used": True,
        "recovery": {
            "model_refit": False,
            "reason": "all seed artifacts completed before package-metadata name shadowing",
        },
        "voxel_selection": {
            **selection,
            "indices_sha256": array_sha256(expected_voxels),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": package_metadata.version("scipy"),
            "sklearn": package_metadata.version("scikit-learn"),
            "cvxopt": getattr(importlib.import_module("cvxopt"), "__version__", "unknown"),
        },
        "seed_runs": seed_runs,
        "pairwise": pairs,
        "decision": {
            "predictive_stability_all_pairs": all(
                pair["predictive_stability_pass"] for pair in pairs
            ),
            "interpretive_instability_any_pair": any(
                pair["interpretive_instability_flag"] for pair in pairs
            ),
        },
    }
    _atomic_json(summary_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = finalize(json.loads(args.config.read_text()), args.output_directory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
