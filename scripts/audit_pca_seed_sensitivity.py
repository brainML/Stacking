#!/usr/bin/env python3
"""Response-free PCA seed sensitivity diagnostics for a feature matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sklearn
from sklearn.decomposition import PCA


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(np.linalg.norm(left, ord="fro"), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(left - right, ord="fro") / denominator)


def run_seed_sensitivity(
    feature_path: Path,
    *,
    seeds: Iterable[int],
    n_components: int,
    svd_solver: str = "auto",
    probe_row_count: int = 256,
) -> dict[str, Any]:
    seed_values = [int(seed) for seed in seeds]
    if len(seed_values) < 2 or len(seed_values) != len(set(seed_values)):
        raise ValueError("provide at least two unique seeds")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if features.ndim < 2:
        raise ValueError("feature array must have a sample axis and feature axes")
    matrix = features.reshape(features.shape[0], -1)
    if not 0 < n_components <= min(matrix.shape):
        raise ValueError("n_components exceeds matrix rank bound")
    if probe_row_count < 1:
        raise ValueError("probe_row_count must be positive")
    probe_count = min(probe_row_count, matrix.shape[0])
    probe_rows = np.unique(
        np.linspace(0, matrix.shape[0] - 1, probe_count, dtype=np.int64)
    )

    fitted: dict[int, dict[str, Any]] = {}
    components: dict[int, np.ndarray] = {}
    probe_grams: dict[int, np.ndarray] = {}
    for seed in seed_values:
        started = time.perf_counter()
        pca = PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            random_state=seed,
        )
        transformed = pca.fit_transform(matrix)
        elapsed = time.perf_counter() - started
        component_matrix = np.asarray(pca.components_, dtype=np.float64)
        probe = np.asarray(transformed[probe_rows], dtype=np.float64)
        probe_gram = probe @ probe.T
        components[seed] = component_matrix
        probe_grams[seed] = probe_gram
        fitted[seed] = {
            "seed": seed,
            "resolved_svd_solver": getattr(pca, "_fit_svd_solver", svd_solver),
            "elapsed_seconds": elapsed,
            "explained_variance_ratio_sum": float(
                np.asarray(pca.explained_variance_ratio_, dtype=np.float64).sum()
            ),
            "components_sha256_float64": array_sha256(component_matrix),
            "probe_transform_sha256_float64": array_sha256(probe),
        }
        del transformed

    pairwise = []
    for left_index, left_seed in enumerate(seed_values):
        for right_seed in seed_values[left_index + 1 :]:
            cross_basis = components[left_seed] @ components[right_seed].T
            singular_values = np.linalg.svd(cross_basis, compute_uv=False)
            # Roundoff in the SVD can place singular values a few ulps outside
            # the mathematical cosine range and bias the chordal distance.
            singular_values = np.clip(singular_values, 0.0, 1.0)
            chordal_squared = max(
                0.0, n_components - float(np.square(singular_values).sum())
            )
            pairwise.append(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "principal_cosine_min": float(singular_values.min()),
                    "principal_cosine_mean": float(singular_values.mean()),
                    "principal_cosine_max": float(singular_values.max()),
                    "subspace_chordal_distance": float(np.sqrt(chordal_squared)),
                    "probe_gram_relative_frobenius_difference": _relative_frobenius(
                        probe_grams[left_seed], probe_grams[right_seed]
                    ),
                }
            )

    return {
        "schema_version": 1,
        "status": "completed",
        "response_data_used": False,
        "feature_shape": list(features.shape),
        "matrix_shape": list(matrix.shape),
        "feature_dtype": str(features.dtype),
        "n_components": n_components,
        "requested_svd_solver": svd_solver,
        "probe_rows": probe_rows.tolist(),
        "runs": [fitted[seed] for seed in seed_values],
        "pairwise": pairwise,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n-components", type=int, required=True)
    parser.add_argument("--svd-solver", default="auto")
    parser.add_argument("--probe-row-count", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_seed_sensitivity(
        args.feature,
        seeds=args.seeds,
        n_components=args.n_components,
        svd_solver=args.svd_solver,
        probe_row_count=args.probe_row_count,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(rendered + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary_path, args.output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
