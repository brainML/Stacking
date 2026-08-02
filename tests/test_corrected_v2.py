import unittest

import numpy as np

from corrected_v2 import (
    FeatureTransform,
    average_acquired_responses,
    fit_corrected_v2,
    ridge_weights,
    solve_simplex_qp,
    validate_nested_splits,
)


def make_nested_fixture():
    rng = np.random.default_rng(41)
    rows = 24
    groups = np.repeat(np.arange(12), 2)
    latent = rng.normal(size=(rows, 5))
    feature_a = np.column_stack([latent[:, :4], np.ones(rows)])
    feature_b = np.column_stack(
        [latent[:, 2], latent[:, 4], latent[:, 0] + 0.05 * rng.normal(size=rows)]
    )
    targets = np.column_stack(
        [
            100.0 + 12.0 * latent[:, 0] - 4.0 * latent[:, 2],
            -30.0 + 0.5 * latent[:, 4] + 0.1 * rng.normal(size=rows),
        ]
    )
    outer = (np.arange(16), np.arange(16, 24))
    inner = tuple(
        (
            np.setdiff1d(np.arange(16), np.arange(start, start + 4)),
            np.arange(start, start + 4),
        )
        for start in range(0, 16, 4)
    )
    return [feature_a, feature_b], targets, groups, outer, inner


class RidgeOracleTests(unittest.TestCase):
    def test_primal_and_dual_are_equivalent_for_rank_deficiency(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=(10, 5))
        features = np.column_stack([base, base[:, 0], np.zeros(10)])
        targets = rng.normal(size=(10, 3))
        primal = ridge_weights(features, targets, 0.3, solver="primal")
        dual = ridge_weights(features, targets, 0.3, solver="dual")
        np.testing.assert_allclose(
            features @ primal, features @ dual, rtol=1e-10, atol=1e-10
        )

    def test_auto_uses_dual_when_features_exceed_rows(self):
        rng = np.random.default_rng(4)
        features = rng.normal(size=(7, 19))
        targets = rng.normal(size=(7, 2))
        auto = ridge_weights(features, targets, 2.0, solver="auto")
        dual = ridge_weights(features, targets, 2.0, solver="dual")
        np.testing.assert_allclose(auto, dual, rtol=0.0, atol=0.0)

    def test_invalid_regularization_and_nonfinite_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            ridge_weights(np.ones((3, 2)), np.ones((3, 1)), 0.0)
        invalid = np.ones((3, 2))
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ridge_weights(invalid, np.ones((3, 1)), 1.0)


class ResponseAvailabilityTests(unittest.TestCase):
    def test_only_acquired_repeats_are_averaged_and_empty_images_are_excluded(self):
        responses = np.array(
            [
                [[1.0], [100.0], [9.0]],
                [[3.0], [200.0], [15.0]],
                [[999.0], [300.0], [21.0]],
            ]
        )
        availability = np.array(
            [[True, False, True], [True, False, True], [False, False, False]]
        )
        averaged = average_acquired_responses(responses, availability)
        np.testing.assert_array_equal(averaged.retained_image_mask, [True, False, True])
        np.testing.assert_array_equal(averaged.acquired_repeat_counts, [2, 0, 2])
        np.testing.assert_allclose(averaged.responses[:, 0], [2.0, 12.0])

    def test_unavailable_nan_is_ignored_but_acquired_nan_is_rejected(self):
        responses = np.array([[[1.0]], [[np.nan]]])
        availability = np.array([[True], [False]])
        np.testing.assert_allclose(
            average_acquired_responses(responses, availability).responses, [[1.0]]
        )
        with self.assertRaisesRegex(ValueError, "acquired responses"):
            average_acquired_responses(responses, np.array([[True], [True]]))


class FeatureTransformTests(unittest.TestCase):
    def test_nested_pca_is_deterministic_and_train_fitted(self):
        rng = np.random.default_rng(8)
        train = rng.normal(size=(12, 7))
        validation = rng.normal(size=(4, 7)) + 500.0
        first = FeatureTransform.fit(train, policy="nested_pca", pca_components=3)
        second = FeatureTransform.fit(train, policy="nested_pca", pca_components=3)
        self.assertEqual(first.output_dimension, 3)
        self.assertEqual(first.identity_sha256(), second.identity_sha256())
        np.testing.assert_array_equal(first.components, second.components)
        np.testing.assert_allclose(first.scaler.mean, train.mean(axis=0))
        self.assertGreater(np.abs(first.transform(validation)).mean(), 100.0)

    def test_invalid_transform_policy_and_dimension_are_rejected(self):
        values = np.ones((5, 3))
        with self.assertRaisesRegex(ValueError, "policy"):
            FeatureTransform.fit(values, policy="unknown", pca_components=None)
        with self.assertRaisesRegex(ValueError, "component count"):
            FeatureTransform.fit(values, policy="nested_pca", pca_components=4)


class NestedSplitTests(unittest.TestCase):
    def test_exact_external_nested_splits_are_frozen(self):
        _, targets, groups, outer, inner = make_nested_fixture()
        frozen_outer, frozen_inner = validate_nested_splits(
            targets.shape[0], outer, inner, groups=groups
        )
        np.testing.assert_array_equal(frozen_outer[0], outer[0])
        for observed, expected in zip(frozen_inner, inner):
            np.testing.assert_array_equal(observed[0], expected[0])
            np.testing.assert_array_equal(observed[1], expected[1])
            self.assertFalse(observed[0].flags.writeable)

    def test_inner_group_leakage_is_rejected(self):
        _, targets, groups, outer, inner = make_nested_fixture()
        leaking = list(inner)
        leaking[0] = (np.arange(1, 16), np.array([0]))
        with self.assertRaisesRegex(ValueError, "group leakage"):
            validate_nested_splits(targets.shape[0], outer, leaking, groups=groups)


class SimplexOracleTests(unittest.TestCase):
    def test_solution_is_feasible_and_reports_kkt_diagnostics(self):
        weights, diagnostics = solve_simplex_qp(np.eye(3))
        np.testing.assert_allclose(weights, np.full(3, 1 / 3), atol=1e-10)
        self.assertTrue(diagnostics.success)
        self.assertLessEqual(diagnostics.sum_error, 1e-10)
        self.assertGreaterEqual(diagnostics.minimum_weight, -1e-10)
        self.assertLessEqual(diagnostics.active_stationarity, 1e-8)

    def test_duplicate_expert_preserves_aggregate_prediction(self):
        errors = np.array(
            [[1.0, -0.4, 0.7, -0.2], [-0.5, 0.8, -0.1, 1.2]],
            dtype=np.float64,
        )
        base, _ = solve_simplex_qp(errors @ errors.T / errors.shape[1])
        duplicated_errors = errors[[0, 0, 1]]
        duplicated, _ = solve_simplex_qp(
            duplicated_errors @ duplicated_errors.T / errors.shape[1]
        )
        np.testing.assert_allclose(
            np.array([duplicated[:2].sum(), duplicated[2]]), base, atol=1e-8
        )
        np.testing.assert_allclose(duplicated @ duplicated_errors, base @ errors, atol=1e-8)


class RepresentationSensitivityTests(unittest.TestCase):
    def test_ridge_prediction_is_invariant_to_orthogonal_rotation(self):
        rng = np.random.default_rng(18)
        features = rng.normal(size=(30, 6))
        targets = rng.normal(size=(30, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        base = features @ ridge_weights(features, targets, 0.4, solver="primal")
        rotated_features = features @ rotation
        rotated = rotated_features @ ridge_weights(
            rotated_features, targets, 0.4, solver="primal"
        )
        np.testing.assert_allclose(base, rotated, rtol=1e-11, atol=1e-11)

    def test_nonorthogonal_reparameterization_is_explicitly_not_invariant(self):
        rng = np.random.default_rng(19)
        features = rng.normal(size=(25, 4))
        targets = rng.normal(size=(25, 2))
        transform = np.eye(4)
        transform[0, 1] = 3.0
        base = features @ ridge_weights(features, targets, 2.0, solver="primal")
        changed_features = features @ transform
        changed = changed_features @ ridge_weights(
            changed_features, targets, 2.0, solver="primal"
        )
        self.assertGreater(np.max(np.abs(base - changed)), 1e-4)


class CorrectedEstimatorTests(unittest.TestCase):
    def _fit(self, features, targets, groups, outer, inner):
        return fit_corrected_v2(
            features,
            targets,
            row_ids=np.array([f"row-{index:02d}" for index in range(targets.shape[0])]),
            target_ids=("voxel-a", "voxel-b"),
            feature_ids=("expert-a", "expert-b"),
            outer_split=outer,
            inner_splits=inner,
            lambdas=np.array([0.01, 0.1, 1.0, 10.0]),
            groups=groups,
        )

    def test_end_to_end_outputs_are_finite_raw_unit_and_deterministic(self):
        features, targets, groups, outer, inner = make_nested_fixture()
        first = self._fit(features, targets, groups, outer, inner)
        second = self._fit(features, targets, groups, outer, inner)
        self.assertEqual(first.mixture_weights.shape, (2, 2))
        self.assertEqual(first.expert_test_predictions.shape, (2, 8, 2))
        self.assertEqual(first.ensemble_oof_prediction.shape, (16, 2))
        self.assertEqual(first.ensemble_test_prediction.shape, (8, 2))
        self.assertTrue(np.isfinite(first.ensemble_test_prediction).all())
        self.assertGreater(first.ensemble_test_prediction[:, 0].mean(), 50.0)
        np.testing.assert_allclose(first.mixture_weights.sum(axis=1), 1.0, atol=1e-10)
        self.assertTrue(all(item.success for item in first.mixture_diagnostics))
        np.testing.assert_array_equal(first.mixture_weights, second.mixture_weights)
        np.testing.assert_array_equal(
            first.ensemble_test_prediction, second.ensemble_test_prediction
        )
        self.assertEqual(first.split_identity_sha256, second.split_identity_sha256)

    def test_outer_test_shift_cannot_change_fitted_parameters(self):
        features, targets, groups, outer, inner = make_nested_fixture()
        baseline = self._fit(features, targets, groups, outer, inner)
        shifted_features = [array.copy() for array in features]
        for array in shifted_features:
            array[outer[1]] += 10000.0
        shifted_targets = targets.copy()
        shifted_targets[outer[1]] -= 5000.0
        shifted = self._fit(shifted_features, shifted_targets, groups, outer, inner)
        np.testing.assert_array_equal(baseline.mixture_weights, shifted.mixture_weights)
        for left, right in zip(baseline.expert_results, shifted.expert_results):
            np.testing.assert_array_equal(left.selected_lambdas, right.selected_lambdas)
            np.testing.assert_array_equal(left.final_weights, right.final_weights)

    def test_nested_pca_policy_is_fold_fitted_and_reproducible(self):
        features, targets, groups, outer, inner = make_nested_fixture()
        kwargs = dict(
            row_ids=np.arange(targets.shape[0]),
            target_ids=("a", "b"),
            feature_ids=("x", "y"),
            outer_split=outer,
            inner_splits=inner,
            lambdas=np.array([0.1, 1.0]),
            groups=groups,
            feature_transform_policy="nested_pca",
            pca_components=2,
        )
        first = fit_corrected_v2(features, targets, **kwargs)
        second = fit_corrected_v2(features, targets, **kwargs)
        np.testing.assert_array_equal(
            first.ensemble_test_prediction, second.ensemble_test_prediction
        )
        self.assertEqual(first.expert_results[0].final_weights.shape, (2, 2))

    def test_column_scaling_is_removed_by_train_fitted_standardization(self):
        features, targets, groups, outer, inner = make_nested_fixture()
        baseline = self._fit(features, targets, groups, outer, inner)
        scaled = [features[0] * np.array([2.0, 0.5, 4.0, 3.0, 9.0]), features[1] * 7.0]
        observed = self._fit(scaled, targets, groups, outer, inner)
        np.testing.assert_allclose(
            baseline.ensemble_test_prediction,
            observed.ensemble_test_prediction,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_pca_dimension_is_a_scientific_parameter(self):
        features, targets, groups, outer, inner = make_nested_fixture()
        common = dict(
            row_ids=np.arange(targets.shape[0]),
            target_ids=("a", "b"),
            feature_ids=("x", "y"),
            outer_split=outer,
            inner_splits=inner,
            lambdas=np.array([0.1, 1.0]),
            groups=groups,
            feature_transform_policy="nested_pca",
        )
        two = fit_corrected_v2(features, targets, pca_components=2, **common)
        three = fit_corrected_v2(features, targets, pca_components=3, **common)
        self.assertGreater(
            np.max(np.abs(two.ensemble_test_prediction - three.ensemble_test_prediction)),
            1e-6,
        )

    def test_duplicate_row_and_feature_identities_are_rejected(self):
        features, targets, groups, outer, inner = make_nested_fixture()
        with self.assertRaisesRegex(ValueError, "row IDs must be unique"):
            fit_corrected_v2(
                features,
                targets,
                row_ids=np.zeros(targets.shape[0]),
                target_ids=("a", "b"),
                feature_ids=("x", "y"),
                outer_split=outer,
                inner_splits=inner,
                lambdas=np.array([1.0]),
                groups=groups,
            )


if __name__ == "__main__":
    unittest.main()
