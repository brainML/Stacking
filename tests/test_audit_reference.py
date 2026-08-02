import unittest

import numpy as np

from audit_reference import Standardizer, signed_r2, validate_splits
from audit_fixture import make_audit_fixture


class TrainFittedPreprocessingTests(unittest.TestCase):
    def test_validation_shift_does_not_change_fitted_statistics(self):
        features, _, _, folds = make_audit_fixture()
        fit, validation = folds[-1]
        scaler = Standardizer.fit(features[fit])

        np.testing.assert_allclose(scaler.mean, features[fit].mean(axis=0))
        np.testing.assert_allclose(scaler.scale, np.where(features[fit].std(axis=0) > 0, features[fit].std(axis=0), 1.0))
        self.assertGreater(np.abs(scaler.transform(features[validation])[:, 0].mean()), 10.0)

    def test_inverse_transform_returns_raw_units(self):
        _, targets, _, folds = make_audit_fixture()
        fit, validation = folds[-1]
        scaler = Standardizer.fit(targets[fit])
        restored = scaler.inverse_transform(scaler.transform(targets[validation]))
        np.testing.assert_allclose(restored, targets[validation], rtol=0.0, atol=1e-12)

    def test_constant_column_has_unit_scale(self):
        features, _, _, folds = make_audit_fixture()
        scaler = Standardizer.fit(features[folds[0][0]])
        self.assertEqual(scaler.scale[-1], 1.0)
        np.testing.assert_allclose(scaler.transform(features)[:, -1], 0.0)


class FrozenSplitTests(unittest.TestCase):
    def test_group_aware_folds_preserve_exact_identity(self):
        features, _, groups, folds = make_audit_fixture()
        frozen = validate_splits(features.shape[0], folds, groups=groups)
        for (expected_fit, expected_validation), (fit, validation) in zip(folds, frozen):
            np.testing.assert_array_equal(fit, expected_fit)
            np.testing.assert_array_equal(validation, expected_validation)
            self.assertFalse(fit.flags.writeable)
            self.assertFalse(validation.flags.writeable)

    def test_group_leakage_is_rejected(self):
        _, _, groups, _ = make_audit_fixture()
        leaking = ((np.arange(1, 12), np.array([0])),)
        with self.assertRaisesRegex(ValueError, "group leakage"):
            validate_splits(12, leaking, groups=groups, require_partition=False)

    def test_validation_rows_must_be_one_pass_partition(self):
        _, _, groups, folds = make_audit_fixture()
        duplicated = (folds[0], folds[1], folds[1])
        with self.assertRaisesRegex(ValueError, "one-pass partition"):
            validate_splits(12, duplicated, groups=groups)


class ScoreSemanticsTests(unittest.TestCase):
    def test_constant_target_score_is_zero(self):
        target = np.ones((6, 2))
        prediction = target.copy()
        np.testing.assert_array_equal(signed_r2(prediction, target), np.zeros(2))

    def test_negative_scores_are_retained(self):
        target = np.arange(8, dtype=float)[:, None]
        prediction = -target
        self.assertLess(signed_r2(prediction, target)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
