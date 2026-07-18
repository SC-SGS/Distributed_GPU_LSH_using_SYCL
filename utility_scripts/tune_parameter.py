#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################
# Authors: Marcel Breyer #
##########################

import argparse
import subprocess
import optuna
from pathlib import Path
import numpy as np
import csv
optuna.logging.set_verbosity(optuna.logging.WARNING)

def float_to_string(value, decimal_places):
    rounded = f"{value:.{decimal_places}f}"
    return rounded.rstrip('0').rstrip('.')

def get_data_set_dimensions(path):
    with open(path, "rb") as f:
        real_type_size = int(np.frombuffer(f.read(np.dtype(index_type).itemsize), dtype=index_type)[0])
        parsing_type_size = int(np.dtype(real_type).itemsize)
        if real_type_size != parsing_type_size:
            raise ValueError(f"The data was stored using a {real_type_size} Byte type but is now read using a {parsing_type_size} Byte type which is not supported!")
        num_data_points = int(np.frombuffer(f.read(np.dtype(index_type).itemsize), dtype=index_type)[0])
        num_dimensions = int(np.frombuffer(f.read(np.dtype(index_type).itemsize), dtype=index_type)[0])
    return num_data_points, num_dimensions

def get_prime_numbers_in_range(a, b):
    """Return all primes in [a, b] using the Sieve of Eratosthenes."""
    sieve = [True] * (b + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(b**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, b + 1, i):
                sieve[j] = False
    return [i for i in range(max(2, a), b + 1) if sieve[i]]

def run_trial(trial, build_path, data_set, hash_function, num_nearest_neighbors, params):
    cmd = [
        f"{build_path}/prog",
        "--hash_function", hash_function,
        "--hash_pool_size", str(params["hash_pool_size"]),
        "--num_hash_functions", str(params["num_hash_functions"]),
        "--num_hash_tables", str(params["num_hash_tables"]),
        "--hash_table_size", str(params["hash_table_size"])
    ]
    if hash_function != "entropy_based":
        cmd.extend(["-w", str(params["w"])])
    if hash_function != "random_projections":
        cmd.extend(["--num_cut_off_points", str(params["num_cut_off_points"])])
    cmd.extend([
        "--indices_ground_truth_file", f"{data_set}_correct_indices.bin",
        "--distances_ground_truth_file", f"{data_set}_correct_distances.bin",
        data_set,
        str(num_nearest_neighbors),
    ])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    # --- parse accuracy from stdout ---
    recall = None
    for line in result.stderr.splitlines():
        if "recall" in line.lower():
            recall = float(line.split(":")[-1].split("%")[0].strip()) / 100.0
        if "error ratio" in line.lower():
            trial.set_user_attr("error_ratio", float(line.split(":")[-1].strip()))
        if "fit the nearest-neighbors estimator" in line.lower():
            trial.set_user_attr("fit_runtime", line.split(" ")[-1].strip()[:-1])
        if "calculated" in line.lower():
            trial.set_user_attr("search_runtime", line.split(" ")[-1].strip()[:-1])
        if "total runtime" in line.lower():
            trial.set_user_attr("total_runtime", line.split(":")[-1].strip())

    # None if not found
    return recall

def objective(trial, build_path, data_set, hash_function, hash_table_size, num_nearest_neighbors):
    params = {
        "hash_pool_size": trial.suggest_int("hash_pool_size", 256, 4096),
        "num_hash_functions": trial.suggest_int("num_hash_functions", 4, 512),
        "num_hash_tables": trial.suggest_int("num_hash_tables", 4, 512),
        "hash_table_size": trial.suggest_categorical("hash_table_size", hash_table_size),
    }
    if hash_function != "entropy_based":
        params["w"] = trial.suggest_float("w", 0.1, 10.0, step=0.1)
    if hash_function == "entropy_based":
        params["num_cut_off_points"] = trial.suggest_int("num_cut_off_points", 6, 128)
    if hash_function == "mixed_hash_functions":
        num_data_points, _ = get_data_set_dimensions(data_set)
        params["num_cut_off_points"] = trial.suggest_int("num_cut_off_points", int(num_data_points / 4096), int(num_data_points / 64))

    accuracy = run_trial(trial, build_path, data_set, hash_function, num_nearest_neighbors, params)

    if accuracy is None:
        raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()

    hash_function_choices = ["random_projections", "entropy_based", "mixed_hash_functions"]

    parser.add_argument("--build_path", type=str, required=True)
    parser.add_argument("--data_set", type=str, required=True)
    parser.add_argument("--use_64bit_real_type", help="use 64bit real_type instead of 32bit", action="store_true")
    parser.add_argument("--knn", type=int, required=True, help="number of nearest neighbors")
    parser.add_argument("--hash_function", nargs="+", help="the hash functions", choices=hash_function_choices, default=hash_function_choices)
    parser.add_argument("--n_trials", type=int, default=100, help="number of tuning trials")
    parser.add_argument("--top_n", type=int, default=5, help="the top n trials to output at te end")

    args = parser.parse_args()

    # set the used real_type and index_type
    if args.use_64bit_real_type:
        real_type = np.float64
    else:
        real_type = np.float32
    index_type = np.uint64

    # pre-compute possible hash table sizes -> prime numbers
    hash_table_sizes = get_prime_numbers_in_range(1000, 200000)

    tuning_results = []

    for used_hash_function in args.hash_function:
        print(f"Tuning {args.knn} nearest-neighbors using {used_hash_function}:")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),  # Bayesian optimization
            # sampler=optuna.samplers.RandomSampler(),
            pruner=optuna.pruners.MedianPruner(),
        )

        study.optimize(
            lambda trial: objective(trial, args.build_path, args.data_set, used_hash_function, hash_table_sizes, args.knn),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        # filter out None values (i.e., trials that were not successful)
        successful_trials = [trial for trial in study.trials if trial.value is not None]
        # calculate top_n: the minimum of the provided top_n, number of successful trials
        top_n = min(int(args.top_n), len(successful_trials))
        # sort the successful trials
        successful_trials_sorted = sorted(successful_trials, key=lambda t: t.value, reverse=True)
        top_trials = successful_trials_sorted[:top_n]

        print(f"\n--- Top {top_n} Trials ({len(successful_trials)} out of {len(study.trials)} trials succeeded) ---")
        for rank, trial in enumerate(top_trials, start=1):
            print(f"\n[{rank}] accuracy={float_to_string(trial.value * 100, 2)}%  error ratio={trial.user_attrs['error_ratio']}  runtime={trial.user_attrs['total_runtime']}")
            for k, v in trial.params.items():
                print(f"  {k}: {float_to_string(v, 1)}")

        # later, output ALL results to the .csv file (sorted by accuracy)
        for rank, trial in enumerate(successful_trials_sorted, start=1):
            trial_results = {}
            trial_results["data_set"] = Path(args.data_set).name
            trial_results["hash_function"] = used_hash_function
            trial_results["accuracy"] = float_to_string(trial.value * 100, 2) + "%"
            trial_results["error_ratio"] = trial.user_attrs['error_ratio']
            trial_results["fit_runtime"] = trial.user_attrs['fit_runtime']
            trial_results["search_runtime"] = trial.user_attrs['search_runtime']
            trial_results["total_runtime"] = trial.user_attrs['total_runtime']
            for k, v in trial.params.items():
                trial_results[k] = float_to_string(v, 1)

            tuning_results.append(trial_results)

    # get all dict keys and create the final csv file -> preserve the key order
    seen = {}
    for d in tuning_results:
        for key in d.keys():
            seen[key] = None  # dict preserves insertion order, duplicates are ignored
    all_keys = list(seen.keys())
    with open(f"tuning_results_{Path(args.data_set).name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore', restval='')
        writer.writeheader()
        writer.writerows(tuning_results)