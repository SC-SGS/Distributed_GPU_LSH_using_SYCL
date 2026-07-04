import argparse
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import warnings

def parse_metadata(filename, comments='#'):
    internal_metadata = {}
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith(comments):
                break
            key, value = line[2:].strip().split(': ')
            internal_metadata[key] = value
    return internal_metadata

def plot_hash_value_distribution(row, file):
    metadata = parse_metadata(file)
    fig.suptitle(f"hash function: {metadata['hash_function_type']} | num data points: {metadata['num_data_points']}")

    ax1, ax2 = axes[row]

    distributions = genfromtxt(file, delimiter=',', comments='#', ndmin=2)

    for dist in distributions:
        data = dist[1:]
        label = f"hash table {int(dist[0])}"

        ax1.plot(np.arange(len(data)), data, alpha=0.2, label=label)

        unique, freq = np.unique(data, return_counts=True)
        ax2.plot(unique, freq, alpha=0.2, label=label)

    # calculate and plot the mean over all distributions
    all_data = distributions[:, 1:]
    mean_data = np.mean(all_data, axis=0)
    ax1.plot(np.arange(len(mean_data)), mean_data, color='red', label='mean')

    max_count = int(all_data.max()) + 1
    all_counts = np.array([np.bincount(d.astype(int), minlength=max_count) for d in all_data])

    # average only over non-zero entries per bucket size
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        mean_freq = np.where(all_counts > 0, all_counts, np.nan)
        mean_freq = np.nanmean(mean_freq, axis=0)

    nonzero = ~np.isnan(mean_freq)
    ax2.plot(np.where(nonzero)[0], mean_freq[nonzero], color='red', label='mean')

    ax1.set_yscale("log")
    ax1.set_xlabel("hash value")
    ax1.set_ylabel("count")
    ax1.set_title("Count per Hash Value")

    ax2.set_yscale("log")
    ax2.set_xlabel("hash bucket size")
    ax2.set_ylabel("number of buckets with this size")
    ax2.set_title("Distribution of Bucket Sizes")

def plot_nearest_neighbor_search_count_distribution(row, file):
    metadata = parse_metadata(args.nn_count_file)
    fig.suptitle(f"hash function: {metadata['hash_function_type']} | num data points: {metadata['num_data_points']}")

    ax1, ax2 = axes[row]

    counts = genfromtxt(file, delimiter=',', comments='#')

    for count in counts:
        data = count[1:]
        label = f"hash table {int(count[0])}"

        ax1.plot(np.arange(len(data)), data, alpha=0.2, label=label)

        unique, freq = np.unique(data, return_counts=True)
        ax2.plot(unique, freq, alpha=0.2, label=label)

    # calculate and plot the mean over all counts
    all_data = counts[:, 1:]
    mean_data = np.mean(all_data, axis=0)
    ax1.plot(np.arange(len(mean_data)), mean_data, color='red', label='mean')

    max_count = int(all_data.max()) + 1
    all_counts = np.array([np.bincount(d.astype(int), minlength=max_count) for d in all_data])

    # average only over non-zero entries per bucket size
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        mean_freq = np.where(all_counts > 0, all_counts, np.nan)
        mean_freq = np.nanmean(mean_freq, axis=0)

    nonzero = ~np.isnan(mean_freq)
    ax2.plot(np.where(nonzero)[0], mean_freq[nonzero], color='red', label='mean')

    ax1.set_yscale("log")
    ax1.set_xlabel("query data point")
    ax1.set_ylabel("count")
    ax1.set_title("Count per Query Data Point")

    ax2.set_yscale("log")
    ax2.set_xlabel("nearest-neighbor calculation count")
    ax2.set_ylabel("number of query data points with this count")
    ax2.set_title("Distribution of Nearest-Neighbor Calculation Counts")

def plot_total_nearest_neighbor_search_count_distribution(row, file):
    metadata = parse_metadata(args.nn_count_file)
    fig.suptitle(f"hash function: {metadata['hash_function_type']} | num data points: {metadata['num_data_points']}")

    ax1, ax2 = axes[row]

    counts = genfromtxt(file, delimiter=',', comments='#')

    all_data = counts[:, 1:]
    sum_data = np.sum(all_data, axis=0)
    ax1.plot(np.arange(len(sum_data)), sum_data, color='red', label='sum')

    max_count = int(sum_data.max()) + 1
    counts_distribution = np.array(np.bincount(sum_data.astype(int), minlength=max_count))
    nonzero = counts_distribution > 0
    ax2.plot(np.where(nonzero)[0], counts_distribution[nonzero], color='red', label='sum')

    ax1.set_yscale("log")
    ax1.set_xlabel("query data point")
    ax1.set_ylabel("total count")
    ax1.set_title("Total count per Query Data Point")

    ax2.set_yscale("log")
    ax2.set_xlabel("nearest-neighbor calculation count")
    ax2.set_ylabel("number of query data points with this count")
    ax2.set_title("Total Distribution of Nearest-Neighbor Calculation Counts")


if __name__ == "__main__":
    # setup command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution_file", type=str, required=False, help="input hash table distribution file")
    parser.add_argument("--nn_count_file", type=str, required=False, help="input nearest-neighbor calculation count file")
    parser.add_argument("--output_file", type=str, required=False, help="output plot file")
    args = parser.parse_args()

    # determine how many rows in the final plot we need
    num_rows = 0
    if args.distribution_file:
        num_rows = num_rows + 1 # per hash table
    if args.nn_count_file:
        num_rows = num_rows + 2 # per hash table + sum
    # at least one row must be there
    if num_rows == 0:
        raise ValueError("At least one of --distribution_file or --nn_count_file must be provided.")

    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 5 * num_rows), squeeze=False)

    current_row = 0

    if args.distribution_file is not None:
        plot_hash_value_distribution(current_row, args.distribution_file)
        current_row += 1

    if args.nn_count_file is not None:
        plot_nearest_neighbor_search_count_distribution(current_row, args.nn_count_file)
        current_row += 1
        plot_total_nearest_neighbor_search_count_distribution(current_row, args.nn_count_file)

    plt.tight_layout()

    if args.output_file is not None:
        plt.savefig(args.output_file, dpi=150, bbox_inches="tight")
    else:
        plt.show()