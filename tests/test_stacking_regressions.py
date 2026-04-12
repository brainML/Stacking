import unittest
from unittest import mock

import numpy as np

import concatenate
import stacking


class FeatRidgeCVTests(unittest.TestCase):
    def test_simple_ridge_does_not_call_cross_val_ridge_for_final_refit(self):
        rng = np.random.default_rng(0)
        train_features = rng.normal(size=(12, 4))
        train_targets = rng.normal(size=(12, 3))
        test_features = rng.normal(size=(5, 4))

        def fail_cross_val(*args, **kwargs):
            raise AssertionError("cross_val_ridge should not be used for simple_ridge")

        with mock.patch.object(stacking, "cross_val_ridge", side_effect=fail_cross_val):
            train_preds, train_err, test_preds, train_scores, train_variances = stacking.feat_ridge_CV(
                train_features,
                train_targets,
                test_features,
                method="simple_ridge",
                n_folds=3,
            )

        expected_weights = stacking.ridge(train_features, train_targets, 100)
        expected_test_preds = test_features @ expected_weights

        self.assertEqual(train_preds.shape, train_targets.shape)
        self.assertEqual(train_err.shape, train_targets.shape)
        self.assertEqual(train_scores.shape, (train_targets.shape[1],))
        self.assertEqual(train_variances.shape, (train_targets.shape[1],))
        np.testing.assert_allclose(test_preds, expected_test_preds)


class UtilityTests(unittest.TestCase):
    def test_get_cv_indices_covers_all_folds(self):
        indices = stacking.get_cv_indices(11, 3)
        expected = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])

        self.assertEqual(indices.dtype, np.intp)
        np.testing.assert_array_equal(indices, expected)


class StackingFallbackTests(unittest.TestCase):
    def test_stacking_fallback_without_cvxopt_returns_simplex_weights(self):
        rng = np.random.default_rng(1)
        train_data = rng.normal(size=(18, 5))
        test_data = rng.normal(size=(6, 5))
        train_features = [rng.normal(size=(18, 4)), rng.normal(size=(18, 3))]
        test_features = [rng.normal(size=(6, 4)), rng.normal(size=(6, 3))]

        with mock.patch.object(stacking, "solvers", None), mock.patch.object(
            stacking, "matrix", None
        ):
            result = stacking.stacking_fmri(
                train_data,
                test_data,
                train_features,
                test_features,
            )

        weights = result[5]
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= -1e-8))
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)

    def test_stacking_cv_fallback_returns_finite_outputs_with_expected_shapes(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(24, 6))
        features = [rng.normal(size=(24, 5)), rng.normal(size=(24, 4))]

        with mock.patch.object(stacking, "solvers", None), mock.patch.object(
            stacking, "matrix", None
        ):
            r2s, stacked_r2s, r2s_weighted, r2s_train, stacked_train, weights = (
                stacking.stacking_CV_fmri(data, features, n_folds=3)
            )

        self.assertEqual(r2s.shape, (2, 6))
        self.assertEqual(stacked_r2s.shape, (6,))
        self.assertEqual(r2s_weighted.shape, (2, 6))
        self.assertEqual(r2s_train.shape, (2, 6))
        self.assertEqual(stacked_train.shape, (6,))
        self.assertEqual(weights.shape, (6, 2))
        self.assertTrue(np.all(np.isfinite(r2s)))
        self.assertTrue(np.all(np.isfinite(stacked_r2s)))
        self.assertTrue(np.all(np.isfinite(weights)))
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)


class ConcatenateTests(unittest.TestCase):
    def test_concatenate_cv_returns_finite_scores(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(24, 7))
        features = [rng.normal(size=(24, 5)), rng.normal(size=(24, 2))]

        (scores,) = concatenate.concatenate_CV_fmri(data, features, n_folds=3)

        self.assertEqual(scores.shape, (7,))
        self.assertTrue(np.all(np.isfinite(scores)))


if __name__ == "__main__":
    unittest.main()
