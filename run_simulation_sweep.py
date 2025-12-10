#!/usr/bin/env python

import argparse
import itertools
import numpy as np
import pandas as pd

from simulation_experiment import run_one_simulation

# Number of samples (n)
N_LIST = [
    50,100,200,400
]

# Feature dimensions for each of the 4 feature spaces
DS_LIST = [
    [10, 10, 10, 10],
    [10, 100, 100, 100],
    [100,100,100,100]
]

# Alpha for each of the 4 feature spaces
ALPHA_LIST = [
    [0.3, 0.3, 0.3, 0.1], 
    [0.5, 0.2, 0.2, 0.1],
     [0.7, 0.1, 0.1, 0.1],
]

# Output noise (y_noise / sigma)
NOISE_LIST = [
    0, 0.5, 1, 1.5
]

# Correlation between feature spaces
CORREL_LIST = [
     0, 0.1, 0.2, 0.3
]

# Fixed parameters (you can change these too if you want)
SCALE = 0.5
Y_DIM = 2

# How many Monte Carlo runs per parameter setting?
N_RUNS_PER_SETTING = 50


def build_tasks():
    """
    Flatten all (parameter combination × run) into a list of tasks.
    Each task gets a unique global_index so you can shard across machines.
    """
    tasks = []
    idx = 0

    for n, ds, alpha, noise, correl in itertools.product(
        N_LIST, DS_LIST, ALPHA_LIST, NOISE_LIST, CORREL_LIST
    ):
        for run_id in range(N_RUNS_PER_SETTING):
            tasks.append(
                dict(
                    global_index=idx,
                    n=int(n),
                    ds=list(ds),
                    alpha=list(alpha),
                    noise=float(noise),
                    correl=float(correl),
                    run=run_id,
                )
            )
            idx += 1

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Run a slice of the parameter sweep for the stacking vs concatenation simulations."
    )
    parser.add_argument(
        "--start-index",
        type=int,
        required=True,
        help="First task index (inclusive) to run.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        required=True,
        help="Last task index (inclusive) to run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle file name, e.g. results_0_99.pkl",
    )
    parser.add_argument(
        "--print-task-count",
        action="store_true",
        help="If set, only print total number of tasks and exit.",
    )

    args = parser.parse_args()

    tasks = build_tasks()
    total_tasks = len(tasks)

    if args.print_task_count:
        print(f"Total tasks: {total_tasks}")
        return

    if total_tasks == 0:
        raise ValueError("No tasks were generated. Did you fill in the parameter lists?")

    start = max(0, args.start_index)
    end = min(total_tasks - 1, args.end_index)

    if start > end:
        raise ValueError(
            f"Invalid range: start_index={args.start_index}, end_index={args.end_index}, total_tasks={total_tasks}"
        )

    selected_tasks = [t for t in tasks if start <= t["global_index"] <= end]

    print(
        f"Total tasks: {total_tasks}. "
        f"Running tasks {start}..{end} ({len(selected_tasks)} tasks) "
        f"into {args.output}"
    )

    all_dfs = []

    for t in selected_tasks:
        idx = t["global_index"]
        n = t["n"]
        ds = t["ds"]
        alpha = t["alpha"]
        noise = t["noise"]
        correl = t["correl"]
        run_id = t["run"]

        print(
            f"\n=== Task {idx} ===\n"
            f"n={n}, ds={ds}, alpha={alpha}, noise={noise}, correl={correl}, run={run_id}"
        )

        # For reproducibility, seed based on global task index
        np.random.seed(idx)

        df = run_one_simulation(
            samples=n,
            ds=ds,
            scale=SCALE,
            correl=correl,
            alpha=alpha,
            y_dim=Y_DIM,
            y_noise=noise,
        )

        # Attach parameter metadata to each row in this simulation
        df["n"] = n
        df["d1"], df["d2"], df["d3"], df["d4"] = ds
        df["alpha1"], df["alpha2"], df["alpha3"], df["alpha4"] = alpha
        df["sigma"] = noise
        df["correl"] = correl
        df["run"] = run_id
        df["task_index"] = idx

        all_dfs.append(df)

    if not all_dfs:
        print("No tasks were selected; nothing to save.")
        return

    result = pd.concat(all_dfs, ignore_index=True)
    result.to_pickle(args.output)
    print(f"\nSaved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
