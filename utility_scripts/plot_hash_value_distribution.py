import argparse
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

def parse_metadata(filename, comments='#'):
    internal_metadata = {}
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith(comments):
                break
            # strip "# " prefix and split on ": "
            key, value = line[2:].strip().split(': ')
            internal_metadata[key] = value
    return internal_metadata

if __name__ == "__main__":
    # setup command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="input distribution file")
    parser.add_argument("--output_file", type=str, required=False, help="output plot file")
    args = parser.parse_args()

    # parse the metadata
    metadata = parse_metadata(args.input_file)

    # load the data
    distributions = genfromtxt(args.input_file, delimiter=',', comments='#', ndmin=2)

    # create two axes besides each other
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"hash function: {metadata['hash_function_type']} | num data points: {metadata['num_data_points']}")

    for dist in distributions:
        data = dist[1:]
        label = f"hash table {int(dist[0])}"

        # left plot: count per hash value
        ax1.plot(np.arange(len(data)), data, label=label)

        # right plot: distribution of bucket sizes (how often does each count appear)
        unique, freq = np.unique(data, return_counts=True)
        ax2.plot(unique, freq, label=label)

    ax1.set_yscale("log")
    ax1.set_xlabel("hash value")
    ax1.set_ylabel("count")
    ax1.set_title("Count per Hash Value")

    ax2.set_yscale("log")
    ax2.set_xlabel("hash bucket size")
    ax2.set_ylabel("number of buckets with this size")
    ax2.set_title("Distribution of Bucket Sizes")

    plt.tight_layout()

    if args.output_file is not None:
        plt.savefig(args.output_file, dpi=150, bbox_inches="tight")
    else:
        plt.show()