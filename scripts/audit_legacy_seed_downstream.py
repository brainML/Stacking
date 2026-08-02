#!/usr/bin/env python3
"""Bounded downstream audit of unresolved historical PCA randomness."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import tempfile
import time
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def linspace_indices(start: int, stop: int, count: int) -> np.ndarray:
    if start < 0 or stop <= start or count < 1 or count > stop - start:
        raise ValueError("invalid linspace index specification")
    indices = np.unique(np.linspace(start, stop - 1, count, dtype=np.int64))
    if indices.size != count:
        raise ValueError("linspace policy did not produce the requested count")
    return indices


def load_legacy_module(path: Path, expected_sha256: str) -> ModuleType:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError("historical source SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("stacking_audit_legacy", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not create historical source module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_array(path: Path, artifact: dict[str, Any]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing logical input {artifact['logical_id']}")
    if path.stat().st_size != artifact["bytes"]:
        raise ValueError(f"byte-size mismatch for {artifact['logical_id']}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != artifact["shape"] or str(array.dtype) != artifact["dtype"]:
        raise ValueError(f"array-header mismatch for {artifact['logical_id']}")
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {artifact['logical_id']}")
    return array


def column_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("prediction matrices must have the same 2-D shape")
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.square(left_centered).sum(axis=0)
        * np.square(right_centered).sum(axis=0)
    )
    correlations = np.full(left.shape[1], np.nan, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).tiny
    correlations[valid] = (
        left_centered[:, valid] * right_centered[:, valid]
    ).sum(axis=0) / denominator[valid]
    return np.clip(correlations, -1.0, 1.0)


def summarize_pair(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, Any]:
    prediction_corr = column_correlations(
        left["stacked_prediction"], right["stacked_prediction"]
    )
    finite_corr = prediction_corr[np.isfinite(prediction_corr)]
    if finite_corr.size == 0:
        raise ValueError("no finite voxel prediction correlations")
    score_delta = np.abs(left["stacked_r2_raw"] - right["stacked_r2_raw"])
    weight_l1 = np.abs(left["weights"] - right["weights"]).sum(axis=1)
    top_agreement = np.mean(
        np.argmax(left["weights"], axis=1) == np.argmax(right["weights"], axis=1)
    )
    return {
        "median_voxel_prediction_correlation": float(np.median(finite_corr)),
        "p05_voxel_prediction_correlation": float(np.percentile(finite_corr, 5)),
        "finite_prediction_correlation_count": int(finite_corr.size),
        "median_absolute_stacked_signed_r2_delta": float(np.median(score_delta)),
        "p95_absolute_stacked_signed_r2_delta": float(np.percentile(score_delta, 95)),
        "max_absolute_stacked_signed_r2_delta": float(score_delta.max()),
        "median_weight_l1_delta": float(np.median(weight_l1)),
        "p95_weight_l1_delta": float(np.percentile(weight_l1, 95)),
        "top_weight_expert_agreement": float(top_agreement),
    }


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".json.tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _gate_passes(pair: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return bool(
        pair["median_voxel_prediction_correlation"]
        >= thresholds["median_voxel_prediction_correlation_min"]
        and pair["p05_voxel_prediction_correlation"]
        >= thresholds["p05_voxel_prediction_correlation_min"]
        and pair["median_absolute_stacked_signed_r2_delta"]
        <= thresholds["median_absolute_stacked_signed_r2_delta_max"]
        and pair["p95_absolute_stacked_signed_r2_delta"]
        <= thresholds["p95_absolute_stacked_signed_r2_delta_max"]
    )


def run_seed(
    seed: int,
    *,
    feature_arrays: list[np.ndarray],
    response: np.ndarray,
    voxel_indices: np.ndarray,
    legacy: ModuleType,
    n_components: int,
    outer_folds: int,
    outer_fold: int,
    inner_folds: int,
    lambdas: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    np.random.seed(seed)
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=None)
    features = []
    pca_hashes = []
    for array in feature_arrays:
        transformed = pca.fit_transform(np.asarray(array).squeeze())
        features.append(transformed)
        pca_hashes.append(array_sha256(np.asarray(transformed)))

    data = np.asarray(response[:, voxel_indices], dtype=np.float64)
    fold_ids = legacy.CV_ind(data.shape[0], outer_folds)
    train_mask = fold_ids != outer_fold
    test_mask = fold_ids == outer_fold
    train_data = np.nan_to_num(legacy.zscore(data[train_mask]))
    test_data_z = np.nan_to_num(legacy.zscore(data[test_mask]))
    train_features = [
        np.nan_to_num(legacy.zscore(array[train_mask])) for array in features
    ]
    test_features = [
        np.nan_to_num(legacy.zscore(array[test_mask])) for array in features
    ]

    expert_count = len(features)
    voxel_count = voxel_indices.size
    test_count = int(test_mask.sum())
    predictions_train: dict[int, np.ndarray] = {}
    errors: dict[int, np.ndarray] = {}
    predictions_test = np.zeros((expert_count, data.shape[0], voxel_count))
    chosen_lambdas = np.zeros((expert_count, voxel_count))
    train_r2 = np.zeros((expert_count, voxel_count))
    for expert in range(expert_count):
        weights, selected = legacy.cross_val_ridge(
            train_features[expert],
            train_data,
            n_splits=inner_folds,
            lambdas=lambdas,
            do_plot=False,
        )
        chosen_lambdas[expert] = selected
        predictions_train[expert] = train_features[expert] @ weights
        errors[expert] = train_data - predictions_train[expert]
        predictions_test[expert, test_mask] = test_features[expert] @ weights
        train_r2[expert] = legacy.R2(predictions_train[expert], train_data)

    stacked_prediction_all = np.zeros_like(data)
    stacked_train_folds = np.zeros((outer_folds, voxel_count))
    weight_accumulator = np.zeros((voxel_count, expert_count))
    stacked_prediction_all, stacked_train_folds, _, weights = legacy.stacked_core(
        voxel_count,
        range(expert_count),
        errors,
        train_data,
        predictions_test,
        predictions_train,
        test_mask,
        outer_fold,
        stacked_prediction_all,
        stacked_train_folds,
        weight_accumulator,
    )
    stacked_prediction = stacked_prediction_all[test_mask]
    expert_prediction = predictions_test[:, test_mask]
    data_raw_test = data[test_mask]
    arrays = {
        "voxel_indices": voxel_indices,
        "test_rows": np.flatnonzero(test_mask),
        "pca_transform_hashes_ascii": np.asarray(pca_hashes, dtype="S64"),
        "chosen_lambdas": chosen_lambdas,
        "weights": weights,
        "expert_r2_raw": np.asarray(
            [legacy.R2(expert_prediction[i], data_raw_test) for i in range(expert_count)]
        ),
        "expert_r2_z": np.asarray(
            [legacy.R2(expert_prediction[i], test_data_z) for i in range(expert_count)]
        ),
        "train_r2": train_r2,
        "stacked_train_r2": stacked_train_folds[outer_fold],
        "stacked_r2_raw": legacy.R2(stacked_prediction, data_raw_test),
        "stacked_r2_z": legacy.R2(stacked_prediction, test_data_z),
        "stacked_prediction": stacked_prediction,
    }
    metadata = {
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
        "train_row_count": int(train_mask.sum()),
        "test_row_count": test_count,
        "voxel_count": int(voxel_count),
        "pca_transform_hashes": pca_hashes,
    }
    return arrays, metadata


def run(
    config: dict[str, Any],
    replay: dict[str, Any],
    path_map: dict[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError("output directory must be fresh")
    if config["state"] != "planned" or config["response_data_used"] is not True:
        raise ValueError("invalid gate configuration state")
    inputs_by_id = {item["logical_id"]: item for item in replay["inputs"]}
    paths = path_map["inputs"]
    response_id = "nsd_legacy/subject_02/older_cortical_average"
    response = validate_array(Path(paths[response_id]), inputs_by_id[response_id])
    feature_arrays = []
    for label in config["feature_order"]:
        logical_id = f"nsd_legacy/subject_02/alexnet_pooled/{label}"
        feature_arrays.append(validate_array(Path(paths[logical_id]), inputs_by_id[logical_id]))
    legacy = load_legacy_module(
        Path(path_map["source_script"]), replay["source_script"]["sha256"]
    )

    selection = config["voxel_selection"]
    voxel_indices = linspace_indices(
        selection["source_chunk_start"],
        selection["source_chunk_stop"],
        selection["count"],
    )
    pca_config = config["pca"]
    outer = config["outer_evaluation"]
    ridge = config["ridge"]
    lambdas = np.asarray([10.0 ** exponent for exponent in ridge["lambda_exponents"]])
    seed_results: dict[int, dict[str, np.ndarray]] = {}
    seed_metadata = []
    output_directory.mkdir(parents=True)
    for seed in pca_config["initial_numpy_seeds"]:
        arrays, metadata = run_seed(
            int(seed),
            feature_arrays=feature_arrays,
            response=response,
            voxel_indices=voxel_indices,
            legacy=legacy,
            n_components=pca_config["n_components"],
            outer_folds=outer["n_folds"],
            outer_fold=outer["fold"],
            inner_folds=ridge["inner_folds"],
            lambdas=lambdas,
        )
        artifact_path = output_directory / f"seed_{seed}.npz"
        _atomic_npz(artifact_path, arrays)
        metadata["private_artifact_sha256"] = sha256_file(artifact_path)
        seed_metadata.append(metadata)
        seed_results[int(seed)] = arrays

    pairs = []
    seeds = [int(seed) for seed in pca_config["initial_numpy_seeds"]]
    predictive_thresholds = config["decision_gates"]["predictive_stability_all_pairs"]
    interpretive_thresholds = config["decision_gates"]["interpretive_instability_flags"]
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            metrics = summarize_pair(seed_results[left_seed], seed_results[right_seed])
            metrics.update(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "predictive_stability_pass": _gate_passes(
                        metrics, predictive_thresholds
                    ),
                    "interpretive_instability_flag": bool(
                        metrics["top_weight_expert_agreement"]
                        < interpretive_thresholds["top_weight_expert_agreement_min"]
                        or metrics["median_weight_l1_delta"]
                        > interpretive_thresholds["median_weight_l1_delta_max"]
                    ),
                }
            )
            pairs.append(metrics)
    result = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": "completed",
        "response_data_used": True,
        "voxel_selection": {
            **selection,
            "indices_sha256": array_sha256(voxel_indices),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": metadata.version("scipy"),
            "sklearn": metadata.version("scikit-learn"),
            "cvxopt": getattr(importlib.import_module("cvxopt"), "__version__", "unknown"),
        },
        "seed_runs": seed_metadata,
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
    _atomic_json(output_directory / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--path-map", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    replay = json.loads(args.replay_manifest.read_text())
    path_map = json.loads(args.path_map.read_text())
    result = run(config, replay, path_map, args.output_directory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
