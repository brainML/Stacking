#!/usr/bin/env python3
"""Run and hash the deterministic corrected_v2 CPU float64 fixture."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

import corrected_v2
from corrected_v2 import (
    array_sha256,
    average_acquired_responses,
    fit_corrected_v2,
    ridge_weights,
)
from scripts.audit_replay_preflight import sha256


def make_fixture(config: dict[str, Any]):
    rng = np.random.default_rng(config["seed"])
    rows = config["rows"]
    latent = rng.normal(size=(rows, 8))
    narrow = np.column_stack([latent[:, :4], np.ones(rows)])
    projection = rng.normal(size=(8, config["feature_dimensions"][1]))
    wide = latent @ projection + 0.01 * rng.normal(size=(rows, projection.shape[1]))
    targets = np.column_stack(
        [
            20.0 + 3.0 * latent[:, 0] - latent[:, 2],
            -5.0 + 0.7 * latent[:, 6] + 0.05 * rng.normal(size=rows),
            np.full(rows, 7.0),
        ]
    )
    groups = np.repeat(np.arange(config["groups"], dtype=np.int64), 2)
    outer_fit = np.arange(config["outer_fit_rows"], dtype=np.int64)
    outer_test = np.arange(config["outer_fit_rows"], rows, dtype=np.int64)
    fold_size = outer_fit.size // config["inner_folds"]
    inner = []
    for fold in range(config["inner_folds"]):
        validation = outer_fit[fold * fold_size : (fold + 1) * fold_size]
        inner.append((np.setdiff1d(outer_fit, validation), validation))
    return [narrow, wide], targets, groups, (outer_fit, outer_test), tuple(inner)


def result_hashes(result: corrected_v2.CorrectedV2Result) -> dict[str, Any]:
    return {
        "selected_lambdas_sha256": array_sha256(
            np.stack([item.selected_lambdas for item in result.expert_results])
        ),
        "expert_oof_prediction_sha256": array_sha256(
            np.stack([item.oof_prediction for item in result.expert_results])
        ),
        "expert_test_prediction_sha256": array_sha256(result.expert_test_predictions),
        "mixture_weights_sha256": array_sha256(result.mixture_weights),
        "ensemble_oof_prediction_sha256": array_sha256(result.ensemble_oof_prediction),
        "ensemble_test_prediction_sha256": array_sha256(result.ensemble_test_prediction),
        "expert_test_scores_sha256": array_sha256(result.expert_test_scores),
        "ensemble_test_score_sha256": array_sha256(result.ensemble_test_score),
    }


def run(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    features, targets, groups, outer, inner = make_fixture(config)
    row_ids = np.asarray([f"fixture-row-{index:03d}" for index in range(config["rows"])])
    common = dict(
        row_ids=row_ids,
        target_ids=("target-signal-a", "target-signal-b", "target-constant"),
        feature_ids=("expert-narrow", "expert-wide"),
        outer_split=outer,
        inner_splits=inner,
        lambdas=np.asarray(config["lambdas"], dtype=np.float64),
        groups=groups,
    )
    arms = []
    all_repeat_equal = True
    all_simplex_pass = True
    for arm in config["arms"]:
        result = fit_corrected_v2(features, targets, **common, **arm)
        repeated = fit_corrected_v2(features, targets, **common, **arm)
        hashes = result_hashes(result)
        repeated_hashes = result_hashes(repeated)
        repeat_equal = hashes == repeated_hashes
        all_repeat_equal &= repeat_equal
        diagnostic_summary = {
            "success_all": all(item.success for item in result.mixture_diagnostics),
            "sum_error_max": max(item.sum_error for item in result.mixture_diagnostics),
            "minimum_weight_min": min(
                item.minimum_weight for item in result.mixture_diagnostics
            ),
            "active_stationarity_max": max(
                item.active_stationarity for item in result.mixture_diagnostics
            ),
            "inactive_dual_violation_max": max(
                item.inactive_dual_violation for item in result.mixture_diagnostics
            ),
            "complementarity_max": max(
                item.complementarity for item in result.mixture_diagnostics
            ),
        }
        all_simplex_pass &= diagnostic_summary["success_all"]
        arms.append(
            {
                **arm,
                "resolved_ridge_solvers": [
                    item.resolved_solver for item in result.expert_results
                ],
                "ensemble_test_scores": result.ensemble_test_score.tolist(),
                "constant_target_score": float(result.ensemble_test_score[-1]),
                "mixture_diagnostics": diagnostic_summary,
                "bitwise_repeat_equal": repeat_equal,
                "identities": {
                    "row": result.row_identity_sha256,
                    "target": result.target_identity_sha256,
                    "split": result.split_identity_sha256,
                },
                "hashes": hashes,
            }
        )

    rng = np.random.default_rng(config["seed"] + 1)
    equivalence_x = rng.normal(size=(11, 17))
    equivalence_y = rng.normal(size=(11, 4))
    primal_prediction = equivalence_x @ ridge_weights(
        equivalence_x, equivalence_y, 0.7, solver="primal"
    )
    dual_prediction = equivalence_x @ ridge_weights(
        equivalence_x, equivalence_y, 0.7, solver="dual"
    )
    primal_dual_max = float(np.max(np.abs(primal_prediction - dual_prediction)))

    repeats = np.arange(3 * 5 * 2, dtype=np.float64).reshape(3, 5, 2)
    availability = np.array(
        [
            [True, True, False, True, False],
            [True, False, False, True, False],
            [False, True, False, False, False],
        ]
    )
    acquired = average_acquired_responses(repeats, availability)
    gates = config["gates"]
    status = "passed" if (
        all_repeat_equal
        and all_simplex_pass
        and primal_dual_max <= gates["primal_dual_prediction_max_abs"]
        and all(arm["constant_target_score"] == gates["constant_target_score"] for arm in arms)
    ) else "failed"
    return {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": status,
        "estimator_contract": "corrected_v2",
        "numerical_mode": "cpu_float64",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": metadata.version("scipy"),
        },
        "provenance": {
            "config_sha256": sha256(config_path),
            "oracle_source_sha256": sha256(Path(corrected_v2.__file__)),
            "runner_source_sha256": sha256(Path(__file__)),
        },
        "primal_dual_prediction_max_abs": primal_dual_max,
        "acquired_average": {
            "retained_image_mask": acquired.retained_image_mask.tolist(),
            "repeat_counts": acquired.acquired_repeat_counts.tolist(),
            "responses_sha256": array_sha256(acquired.responses),
        },
        "arms": arms,
    }


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), args.config)
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        atomic_write(args.output, result)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
