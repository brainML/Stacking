import tempfile
import unittest
from pathlib import Path

import numpy as np

from corrected_v2 import fit_corrected_v2
from corrected_v2_artifacts import (
    save_corrected_v2_artifact,
    validate_corrected_v2_artifact,
)
from corrected_v2_uncertainty import group_bootstrap_signed_r2_difference
from test_corrected_v2 import make_nested_fixture


def fit_fixture():
    features, targets, groups, outer, inner = make_nested_fixture()
    result = fit_corrected_v2(
        features,
        targets,
        row_ids=np.arange(targets.shape[0]),
        target_ids=("a", "b"),
        feature_ids=("x", "y"),
        outer_split=outer,
        inner_splits=inner,
        lambdas=np.array([0.1, 1.0]),
        groups=groups,
    )
    return result, targets[outer[1]], groups[outer[1]]


class CorrectedArtifactTests(unittest.TestCase):
    def test_atomic_artifact_roundtrip_validates_every_array(self):
        result, _, _ = fit_fixture()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result"
            saved = save_corrected_v2_artifact(
                result, destination, provenance={"fixture": "unit-test"}
            )
            validated = validate_corrected_v2_artifact(destination)
            self.assertEqual(saved, validated)
            self.assertEqual(validated["identities"]["feature_ids"], ["x", "y"])
            self.assertGreater(len(validated["arrays"]), 10)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                save_corrected_v2_artifact(result, destination, provenance={})

    def test_array_corruption_is_detected(self):
        result, _, _ = fit_fixture()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result"
            manifest = save_corrected_v2_artifact(result, destination, provenance={})
            record = manifest["arrays"]["ensemble_test_score"]
            path = destination / record["file"]
            values = np.load(path, allow_pickle=False)
            values[0] += 1.0
            np.save(path, values, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_corrected_v2_artifact(destination)


class GroupBootstrapTests(unittest.TestCase):
    def test_group_bootstrap_is_deterministic_and_directional(self):
        result, target, groups = fit_fixture()
        first = result.ensemble_test_prediction
        second = np.repeat(target.mean(axis=0, keepdims=True), target.shape[0], axis=0)
        left = group_bootstrap_signed_r2_difference(
            first, second, target, groups, seed=22, resamples=200
        )
        right = group_bootstrap_signed_r2_difference(
            first, second, target, groups, seed=22, resamples=200
        )
        np.testing.assert_array_equal(left.differences, right.differences)
        self.assertEqual(left.differences.shape, (200, 2))
        self.assertGreater(left.mean[0], 0.0)
        self.assertTrue(np.all(left.lower <= left.upper))

    def test_bootstrap_rejects_rowwise_or_degenerate_groups(self):
        values = np.ones((4, 2))
        with self.assertRaisesRegex(ValueError, "one value"):
            group_bootstrap_signed_r2_difference(
                values, values, values, np.ones(3), seed=0
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            group_bootstrap_signed_r2_difference(
                values, values, values, np.zeros(4), seed=0
            )


if __name__ == "__main__":
    unittest.main()
