"""Atomic artifacts and independent validation for corrected_v2 results."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from corrected_v2 import CorrectedV2Result, array_sha256


def _result_arrays(result: CorrectedV2Result) -> dict[str, np.ndarray]:
    arrays = {
        "mixture_weights": result.mixture_weights,
        "expert_test_predictions": result.expert_test_predictions,
        "ensemble_oof_prediction": result.ensemble_oof_prediction,
        "ensemble_test_prediction": result.ensemble_test_prediction,
        "expert_test_scores": result.expert_test_scores,
        "ensemble_test_score": result.ensemble_test_score,
    }
    for index, expert in enumerate(result.expert_results):
        prefix = f"expert_{index:03d}"
        arrays.update(
            {
                f"{prefix}_selected_lambdas": expert.selected_lambdas,
                f"{prefix}_validation_mse": expert.validation_mse,
                f"{prefix}_oof_prediction": expert.oof_prediction,
                f"{prefix}_test_prediction": expert.test_prediction,
                f"{prefix}_final_weights": expert.final_weights,
            }
        )
    return arrays


def save_corrected_v2_artifact(
    result: CorrectedV2Result,
    destination: Path,
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one fresh result directory and atomically publish it."""

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        array_records = {}
        for name, values in _result_arrays(result).items():
            array = np.asarray(values)
            if not np.isfinite(array).all():
                raise ValueError(f"result array {name} contains non-finite entries")
            path = temporary / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            array_records[name] = {
                "file": path.name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "array_sha256": array_sha256(array),
            }
        manifest = {
            "schema_version": 1,
            "estimator_contract": "corrected_v2",
            "numerical_mode": "cpu_float64",
            "status": "completed",
            "identities": {
                "row_sha256": result.row_identity_sha256,
                "target_sha256": result.target_identity_sha256,
                "split_sha256": result.split_identity_sha256,
                "feature_ids": list(result.feature_ids),
            },
            "experts": [
                {
                    "index": index,
                    "resolved_solver": expert.resolved_solver,
                    "inner_transform_hashes": list(expert.inner_transform_hashes),
                    "final_transform_hash": expert.final_transform_hash,
                }
                for index, expert in enumerate(result.expert_results)
            ],
            "mixture_diagnostics": [
                asdict(diagnostic) for diagnostic in result.mixture_diagnostics
            ],
            "arrays": array_records,
            "provenance": dict(provenance),
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_corrected_v2_artifact(destination: Path) -> dict[str, Any]:
    """Independently validate array headers and content hashes."""

    destination = Path(destination)
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("corrected_v2 manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("estimator_contract") != "corrected_v2"
        or manifest.get("status") != "completed"
    ):
        raise ValueError("invalid corrected_v2 manifest header")
    records = manifest.get("arrays")
    if not isinstance(records, dict) or not records:
        raise ValueError("manifest contains no array records")
    for name, record in records.items():
        path = destination / record["file"]
        if not path.is_file() or path.parent != destination:
            raise ValueError(f"missing or unsafe array file for {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"array header mismatch for {name}")
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite array values for {name}")
        if array_sha256(array) != record["array_sha256"]:
            raise ValueError(f"array hash mismatch for {name}")
    return manifest
