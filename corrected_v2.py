"""Deterministic CPU float64 oracle for the corrected_v2 estimator contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from audit_reference import Standardizer, signed_r2


IndexSplit = Tuple[np.ndarray, np.ndarray]


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class AcquiredAverage:
    responses: np.ndarray
    retained_image_mask: np.ndarray
    acquired_repeat_counts: np.ndarray


def average_acquired_responses(
    repeated_responses: np.ndarray, availability: np.ndarray
) -> AcquiredAverage:
    """Average acquired repeats only and exclude images with no response."""

    values = np.asarray(repeated_responses, dtype=np.float64)
    mask = np.asarray(availability, dtype=bool)
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("responses must be repeat x image x target with matching mask")
    if values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("response array dimensions must be nonempty")
    expanded = mask[:, :, None]
    if not np.isfinite(values[expanded.repeat(values.shape[2], axis=2)]).all():
        raise ValueError("acquired responses contain non-finite entries")
    counts = mask.sum(axis=0, dtype=np.int64)
    retained = counts > 0
    sums = np.where(expanded, values, 0.0).sum(axis=0)
    averaged = sums[retained] / counts[retained, None]
    retained.setflags(write=False)
    counts.setflags(write=False)
    averaged.setflags(write=False)
    return AcquiredAverage(
        responses=averaged,
        retained_image_mask=retained,
        acquired_repeat_counts=counts,
    )


def _finite_2d(values: np.ndarray, *, name: str, rows: Optional[int] = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty finite 2D array")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} row identity does not match targets")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite entries")
    return array


def _indices(values: Iterable[int], *, n_rows: int, name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.int64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 1D index array")
    if np.unique(array).size != array.size:
        raise ValueError(f"{name} contains duplicate rows")
    if np.any(array < 0) or np.any(array >= n_rows):
        raise ValueError(f"{name} contains out-of-range rows")
    array.setflags(write=False)
    return array


def validate_nested_splits(
    n_rows: int,
    outer_split: Tuple[Iterable[int], Iterable[int]],
    inner_splits: Sequence[Tuple[Iterable[int], Iterable[int]]],
    *,
    groups: Optional[np.ndarray] = None,
) -> tuple[IndexSplit, tuple[IndexSplit, ...]]:
    """Freeze one outer split and an exact one-pass inner partition."""

    outer_fit = _indices(outer_split[0], n_rows=n_rows, name="outer_fit")
    outer_test = _indices(outer_split[1], n_rows=n_rows, name="outer_test")
    if np.intersect1d(outer_fit, outer_test).size:
        raise ValueError("outer split has row leakage")
    if not np.array_equal(
        np.sort(np.concatenate([outer_fit, outer_test])), np.arange(n_rows)
    ):
        raise ValueError("outer split must partition all rows")
    if len(inner_splits) < 2:
        raise ValueError("at least two inner splits are required")

    group_array = None
    if groups is not None:
        group_array = np.asarray(groups)
        if group_array.ndim != 1 or group_array.shape[0] != n_rows:
            raise ValueError("groups must contain one value per row")
        if np.intersect1d(
            np.unique(group_array[outer_fit]), np.unique(group_array[outer_test])
        ).size:
            raise ValueError("outer split has group leakage")

    outer_fit_sorted = np.sort(outer_fit)
    validation_counts = np.zeros(n_rows, dtype=np.int64)
    frozen_inner = []
    for fold, (fit_values, validation_values) in enumerate(inner_splits):
        fit = _indices(fit_values, n_rows=n_rows, name=f"inner_fit[{fold}]")
        validation = _indices(
            validation_values, n_rows=n_rows, name=f"inner_validation[{fold}]"
        )
        if np.intersect1d(fit, validation).size:
            raise ValueError(f"inner split {fold} has row leakage")
        if not np.array_equal(
            np.sort(np.concatenate([fit, validation])), outer_fit_sorted
        ):
            raise ValueError(f"inner split {fold} must partition outer fit rows")
        if group_array is not None and np.intersect1d(
            np.unique(group_array[fit]), np.unique(group_array[validation])
        ).size:
            raise ValueError(f"inner split {fold} has group leakage")
        validation_counts[validation] += 1
        frozen_inner.append((fit, validation))
    if not np.all(validation_counts[outer_fit] == 1):
        raise ValueError("inner validation rows must form an exact one-pass partition")
    return (outer_fit, outer_test), tuple(frozen_inner)


def ridge_weights(
    features: np.ndarray,
    targets: np.ndarray,
    regularization: float,
    *,
    solver: str = "auto",
) -> np.ndarray:
    """Solve positive-regularization ridge in primal or dual float64 form."""

    x = _finite_2d(features, name="features")
    y = _finite_2d(targets, name="targets", rows=x.shape[0])
    value = float(regularization)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("regularization must be finite and strictly positive")
    resolved = solver
    if solver == "auto":
        resolved = "primal" if x.shape[1] <= x.shape[0] else "dual"
    if resolved == "primal":
        system = x.T @ x + value * np.eye(x.shape[1], dtype=np.float64)
        return np.linalg.solve(system, x.T @ y)
    if resolved == "dual":
        system = x @ x.T + value * np.eye(x.shape[0], dtype=np.float64)
        return x.T @ np.linalg.solve(system, y)
    raise ValueError("solver must be 'auto', 'primal', or 'dual'")


@dataclass(frozen=True)
class FeatureTransform:
    scaler: Standardizer
    policy: str
    components: Optional[np.ndarray]

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        *,
        policy: str,
        pca_components: Optional[int],
    ) -> "FeatureTransform":
        array = _finite_2d(values, name="feature transform values")
        scaler = Standardizer.fit(array)
        standardized = scaler.transform(array)
        if policy == "standardize":
            if pca_components is not None:
                raise ValueError("pca_components is invalid for standardize policy")
            return cls(scaler=scaler, policy=policy, components=None)
        if policy != "nested_pca":
            raise ValueError("feature transform policy must be standardize or nested_pca")
        if (
            pca_components is None
            or pca_components < 1
            or pca_components > min(standardized.shape)
        ):
            raise ValueError("invalid nested PCA component count")
        _, _, right = np.linalg.svd(standardized, full_matrices=False)
        components = np.asarray(right[:pca_components], dtype=np.float64)
        # Freeze the arbitrary SVD sign using each component's largest loading.
        pivots = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(components.shape[0]), pivots])
        signs[signs == 0] = 1.0
        components *= signs[:, None]
        components.setflags(write=False)
        return cls(scaler=scaler, policy=policy, components=components)

    @property
    def output_dimension(self) -> int:
        return self.scaler.mean.size if self.components is None else self.components.shape[0]

    def transform(self, values: np.ndarray) -> np.ndarray:
        standardized = self.scaler.transform(values)
        if self.components is None:
            return standardized
        return standardized @ self.components.T

    def identity_sha256(self) -> str:
        parts = [self.scaler.mean, self.scaler.scale]
        if self.components is not None:
            parts.append(self.components.ravel())
        return array_sha256(np.concatenate(parts))


def _ridge_by_target_lambda(
    features: np.ndarray,
    targets: np.ndarray,
    selected_lambdas: np.ndarray,
    *,
    solver: str,
) -> np.ndarray:
    weights = np.zeros((features.shape[1], targets.shape[1]), dtype=np.float64)
    for value in np.unique(selected_lambdas):
        selected = selected_lambdas == value
        weights[:, selected] = ridge_weights(
            features, targets[:, selected], float(value), solver=solver
        )
    return weights


@dataclass(frozen=True)
class RidgeExpertResult:
    selected_lambdas: np.ndarray
    validation_mse: np.ndarray
    oof_prediction: np.ndarray
    test_prediction: np.ndarray
    final_weights: np.ndarray
    inner_transform_hashes: tuple[str, ...]
    final_transform_hash: str
    resolved_solver: str


def fit_ridge_expert(
    features: np.ndarray,
    targets: np.ndarray,
    outer_split: IndexSplit,
    inner_splits: Sequence[IndexSplit],
    lambdas: np.ndarray,
    *,
    solver: str = "auto",
    feature_transform_policy: str = "standardize",
    pca_components: Optional[int] = None,
) -> RidgeExpertResult:
    """Fit one leakage-safe nested-CV ridge expert in raw target units."""

    x = _finite_2d(features, name="features")
    y = _finite_2d(targets, name="targets", rows=x.shape[0])
    lambda_values = np.asarray(lambdas, dtype=np.float64)
    if (
        lambda_values.ndim != 1
        or lambda_values.size == 0
        or not np.isfinite(lambda_values).all()
        or np.any(lambda_values <= 0)
        or np.unique(lambda_values).size != lambda_values.size
    ):
        raise ValueError("lambdas must be unique finite positive values")
    outer_fit, outer_test = outer_split
    if solver not in {"auto", "primal", "dual"}:
        raise ValueError("solver must be 'auto', 'primal', or 'dual'")

    validation_mse = np.zeros((lambda_values.size, y.shape[1]), dtype=np.float64)
    validation_counts = np.zeros(y.shape[1], dtype=np.int64)
    transform_hashes = []
    for fit, validation in inner_splits:
        x_transform = FeatureTransform.fit(
            x[fit], policy=feature_transform_policy, pca_components=pca_components
        )
        y_scaler = Standardizer.fit(y[fit])
        x_fit = x_transform.transform(x[fit])
        y_fit = y_scaler.transform(y[fit])
        x_validation = x_transform.transform(x[validation])
        resolved_fold = solver
        if solver == "auto":
            resolved_fold = "primal" if x_fit.shape[1] <= x_fit.shape[0] else "dual"
        transform_hashes.append(
            array_sha256(
                np.concatenate(
                    [
                        np.frombuffer(bytes.fromhex(x_transform.identity_sha256()), dtype=np.uint8),
                        y_scaler.mean,
                        y_scaler.scale,
                    ]
                )
            )
        )
        for index, value in enumerate(lambda_values):
            weights = ridge_weights(x_fit, y_fit, value, solver=resolved_fold)
            prediction = y_scaler.inverse_transform(x_validation @ weights)
            validation_mse[index] += np.square(y[validation] - prediction).sum(axis=0)
        validation_counts += validation.size
    validation_mse /= validation_counts[None, :]
    selected_lambdas = lambda_values[np.argmin(validation_mse, axis=0)]

    oof_full = np.full_like(y, np.nan, dtype=np.float64)
    for fit, validation in inner_splits:
        x_transform = FeatureTransform.fit(
            x[fit], policy=feature_transform_policy, pca_components=pca_components
        )
        y_scaler = Standardizer.fit(y[fit])
        x_fit = x_transform.transform(x[fit])
        resolved_fold = solver
        if solver == "auto":
            resolved_fold = "primal" if x_fit.shape[1] <= x_fit.shape[0] else "dual"
        weights = _ridge_by_target_lambda(
            x_fit,
            y_scaler.transform(y[fit]),
            selected_lambdas,
            solver=resolved_fold,
        )
        oof_full[validation] = y_scaler.inverse_transform(
            x_transform.transform(x[validation]) @ weights
        )
    if not np.isfinite(oof_full[outer_fit]).all():
        raise RuntimeError("inner folds did not produce complete OOF predictions")
    oof_prediction = oof_full[outer_fit].copy()

    x_transform = FeatureTransform.fit(
        x[outer_fit], policy=feature_transform_policy, pca_components=pca_components
    )
    y_scaler = Standardizer.fit(y[outer_fit])
    x_fit = x_transform.transform(x[outer_fit])
    resolved = solver
    if solver == "auto":
        resolved = "primal" if x_fit.shape[1] <= x_fit.shape[0] else "dual"
    final_weights = _ridge_by_target_lambda(
        x_fit,
        y_scaler.transform(y[outer_fit]),
        selected_lambdas,
        solver=resolved,
    )
    test_prediction = y_scaler.inverse_transform(
        x_transform.transform(x[outer_test]) @ final_weights
    )
    final_transform_hash = array_sha256(
        np.concatenate(
            [
                np.frombuffer(bytes.fromhex(x_transform.identity_sha256()), dtype=np.uint8),
                y_scaler.mean,
                y_scaler.scale,
            ]
        )
    )
    for array in (
        selected_lambdas,
        validation_mse,
        oof_prediction,
        test_prediction,
        final_weights,
    ):
        array.setflags(write=False)
    return RidgeExpertResult(
        selected_lambdas=selected_lambdas,
        validation_mse=validation_mse,
        oof_prediction=oof_prediction,
        test_prediction=test_prediction,
        final_weights=final_weights,
        inner_transform_hashes=tuple(transform_hashes),
        final_transform_hash=final_transform_hash,
        resolved_solver=resolved,
    )


@dataclass(frozen=True)
class SimplexDiagnostics:
    success: bool
    objective: float
    sum_error: float
    minimum_weight: float
    active_stationarity: float
    inactive_dual_violation: float
    complementarity: float
    active_count: int


def solve_simplex_qp(
    quadratic: np.ndarray, *, tolerance: float = 1e-10
) -> tuple[np.ndarray, SimplexDiagnostics]:
    """Minimize w'Pw on the simplex by deterministic active-set enumeration."""

    matrix = np.asarray(quadratic, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("quadratic must be a nonempty square matrix")
    if matrix.shape[0] > 12:
        raise ValueError("CPU oracle active-set solver supports at most 12 experts")
    if not np.isfinite(matrix).all():
        raise ValueError("quadratic contains non-finite entries")
    matrix = 0.5 * (matrix + matrix.T)
    eigen_min = float(np.linalg.eigvalsh(matrix).min())
    if eigen_min < -max(tolerance, np.linalg.norm(matrix, ord=2) * 1e-10):
        raise ValueError("quadratic must be positive semidefinite")

    expert_count = matrix.shape[0]
    best_weight = None
    best_objective = np.inf
    for active_count in range(1, expert_count + 1):
        for active_tuple in combinations(range(expert_count), active_count):
            active = np.asarray(active_tuple, dtype=np.int64)
            submatrix = matrix[np.ix_(active, active)]
            kkt = np.block(
                [
                    [submatrix, np.ones((active_count, 1))],
                    [np.ones((1, active_count)), np.zeros((1, 1))],
                ]
            )
            solution = np.linalg.lstsq(
                kkt, np.r_[np.zeros(active_count), 1.0], rcond=None
            )[0][:-1]
            if solution.min() < -tolerance:
                continue
            weight = np.zeros(expert_count, dtype=np.float64)
            weight[active] = np.maximum(solution, 0.0)
            weight /= weight.sum()
            objective = float(weight @ matrix @ weight)
            if objective < best_objective - tolerance:
                best_objective = objective
                best_weight = weight
    if best_weight is None:
        raise RuntimeError("simplex optimizer found no feasible active set")

    weight = best_weight
    gradient = 2.0 * matrix @ weight
    active = weight > tolerance
    multiplier = -float(gradient[active].mean())
    reduced = gradient + multiplier
    active_stationarity = float(np.max(np.abs(reduced[active])))
    inactive_dual_violation = float(
        max(0.0, -float(reduced[~active].min())) if np.any(~active) else 0.0
    )
    complementarity = float(np.max(np.abs(weight * reduced)))
    sum_error = abs(float(weight.sum()) - 1.0)
    minimum_weight = float(weight.min())
    success = bool(
        sum_error <= tolerance * 10
        and minimum_weight >= -tolerance
        and active_stationarity <= tolerance * 100
        and inactive_dual_violation <= tolerance * 100
        and complementarity <= tolerance * 100
    )
    weight.setflags(write=False)
    return weight, SimplexDiagnostics(
        success=success,
        objective=best_objective,
        sum_error=sum_error,
        minimum_weight=minimum_weight,
        active_stationarity=active_stationarity,
        inactive_dual_violation=inactive_dual_violation,
        complementarity=complementarity,
        active_count=int(active.sum()),
    )


@dataclass(frozen=True)
class CorrectedV2Result:
    expert_results: tuple[RidgeExpertResult, ...]
    mixture_weights: np.ndarray
    mixture_diagnostics: tuple[SimplexDiagnostics, ...]
    expert_test_predictions: np.ndarray
    ensemble_oof_prediction: np.ndarray
    ensemble_test_prediction: np.ndarray
    expert_test_scores: np.ndarray
    ensemble_test_score: np.ndarray
    row_identity_sha256: str
    target_identity_sha256: str
    split_identity_sha256: str
    feature_ids: tuple[str, ...]


def fit_corrected_v2(
    feature_spaces: Sequence[np.ndarray],
    targets: np.ndarray,
    *,
    row_ids: np.ndarray,
    target_ids: Sequence[str],
    feature_ids: Sequence[str],
    outer_split: Tuple[Iterable[int], Iterable[int]],
    inner_splits: Sequence[Tuple[Iterable[int], Iterable[int]]],
    lambdas: np.ndarray,
    groups: Optional[np.ndarray] = None,
    ridge_solver: str = "auto",
    feature_transform_policy: str = "standardize",
    pca_components: Optional[int] = None,
) -> CorrectedV2Result:
    """Fit all experts and their simplex mixture under corrected_v2."""

    y = _finite_2d(targets, name="targets")
    if not feature_spaces or len(feature_spaces) != len(feature_ids):
        raise ValueError("feature spaces and unique feature IDs must align")
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("feature IDs must be unique")
    if len(target_ids) != y.shape[1] or len(set(target_ids)) != len(target_ids):
        raise ValueError("target IDs must be unique and match target columns")
    row_identity = np.asarray(row_ids)
    if row_identity.ndim != 1 or row_identity.shape[0] != y.shape[0]:
        raise ValueError("row IDs must contain one value per row")
    if np.unique(row_identity).size != row_identity.size:
        raise ValueError("row IDs must be unique")
    features = [
        _finite_2d(space, name=f"feature[{index}]", rows=y.shape[0])
        for index, space in enumerate(feature_spaces)
    ]
    frozen_outer, frozen_inner = validate_nested_splits(
        y.shape[0], outer_split, inner_splits, groups=groups
    )
    expert_results = tuple(
        fit_ridge_expert(
            space,
            y,
            frozen_outer,
            frozen_inner,
            lambdas,
            solver=ridge_solver,
            feature_transform_policy=feature_transform_policy,
            pca_components=pca_components,
        )
        for space in features
    )
    outer_fit, outer_test = frozen_outer
    expert_oof = np.stack(
        [result.oof_prediction for result in expert_results], axis=0
    )
    expert_test = np.stack(
        [result.test_prediction for result in expert_results], axis=0
    )
    errors = y[outer_fit][None, :, :] - expert_oof
    target_count = y.shape[1]
    mixture_weights = np.zeros((target_count, len(expert_results)), dtype=np.float64)
    diagnostics = []
    for target in range(target_count):
        error = errors[:, :, target]
        quadratic = error @ error.T / outer_fit.size
        weight, diagnostic = solve_simplex_qp(quadratic)
        if not diagnostic.success:
            raise RuntimeError(f"mixture optimizer failed diagnostics for target {target}")
        mixture_weights[target] = weight
        diagnostics.append(diagnostic)
    ensemble_oof = np.einsum("tf,fnt->nt", mixture_weights, expert_oof)
    ensemble_test = np.einsum("tf,fnt->nt", mixture_weights, expert_test)
    expert_scores = np.stack(
        [signed_r2(expert_test[index], y[outer_test]) for index in range(len(features))]
    )
    ensemble_score = signed_r2(ensemble_test, y[outer_test])
    split_payload = np.concatenate(
        [outer_fit, np.array([-1]), outer_test]
        + [np.concatenate([fit, np.array([-1]), validation]) for fit, validation in frozen_inner]
    )
    target_payload = "\0".join(str(value) for value in target_ids).encode("utf-8")
    row_payload = "\0".join(str(value) for value in row_identity.tolist()).encode("utf-8")
    for array in (
        mixture_weights,
        expert_test,
        ensemble_oof,
        ensemble_test,
        expert_scores,
        ensemble_score,
    ):
        array.setflags(write=False)
    return CorrectedV2Result(
        expert_results=expert_results,
        mixture_weights=mixture_weights,
        mixture_diagnostics=tuple(diagnostics),
        expert_test_predictions=expert_test,
        ensemble_oof_prediction=ensemble_oof,
        ensemble_test_prediction=ensemble_test,
        expert_test_scores=expert_scores,
        ensemble_test_score=ensemble_score,
        row_identity_sha256=hashlib.sha256(row_payload).hexdigest(),
        target_identity_sha256=hashlib.sha256(target_payload).hexdigest(),
        split_identity_sha256=array_sha256(split_payload),
        feature_ids=tuple(feature_ids),
    )
