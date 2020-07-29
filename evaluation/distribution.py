# @author Marcel Breyer
# @date 2020-07-29
# @brief Python3 script for printing the hash table bucket distribution.


import argparse
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
import tabulate


# setup command line arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="the distribution file to plot", type=str, required=True)
args = parser.parse_args()


data = np.loadtxt(args.input_file, delimiter=",")
print("shape: ", data.shape)

ncols = 2
nrows = math.ceil(data.shape[0] / ncols)

arr = []

for idx in range(data.shape[0]):
    arr.append([idx, sum(1 for val in data[idx] if val == 0), max(data[idx]), statistics.median(data[idx]), statistics.mode(data[idx])])
    plt.subplot(nrows, ncols, idx + 1)
    plt.plot(data[idx])

print(tabulate.tabulate(arr, headers=["hash_table", "zeros", "max", "median", "mode"], tablefmt="orgtbl"))
plt.show()
