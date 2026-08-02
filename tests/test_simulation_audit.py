import unittest

import numpy as np

from run_simulation_sweep import (
    ALPHA_LIST,
    CORREL_LIST,
    DS_LIST,
    NOISE_LIST,
    N_LIST,
    N_RUNS_PER_SETTING,
    Y_DIM,
    build_tasks,
)
from simulation_experiment import sample_all_at_once


class ExtendedSweepContractTests(unittest.TestCase):
    def test_executable_task_count_is_explicit(self):
        expected_settings = (
            len(N_LIST)
            * len(DS_LIST)
            * len(ALPHA_LIST)
            * len(NOISE_LIST)
            * len(CORREL_LIST)
        )
        tasks = build_tasks()
        self.assertEqual(expected_settings, 576)
        self.assertEqual(N_RUNS_PER_SETTING, 50)
        self.assertEqual(len(tasks), expected_settings * N_RUNS_PER_SETTING)
        self.assertEqual(len(tasks) * Y_DIM, 57600)
        self.assertEqual([task["global_index"] for task in tasks], list(range(len(tasks))))

    def test_current_mixing_parameter_changes_generated_arrays(self):
        arguments = {
            "n": 40,
            "ds": [3, 3, 3],
            "scale": 0.5,
            "alpha": [0.5, 0.3, 0.2],
            "data_dim": 2,
            "y_noise": 0.0,
        }
        np.random.seed(20260802)
        zero_features, zero_target, _ = sample_all_at_once(correl=0.0, **arguments)
        np.random.seed(20260802)
        mixed_features, mixed_target, _ = sample_all_at_once(correl=0.2, **arguments)

        self.assertFalse(np.allclose(zero_target, mixed_target))
        self.assertTrue(any(not np.allclose(a, b) for a, b in zip(zero_features, mixed_features)))


if __name__ == "__main__":
    unittest.main()
