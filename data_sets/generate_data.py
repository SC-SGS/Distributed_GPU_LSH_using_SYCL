# @author Marcel Breyer
# @date 2020-06-16
# @brief Python3 script for generating data sets.


import argparse
import sklearn.datasets
import sklearn.preprocessing
import numpy as np
import sys
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def size_in_bytes(numpy_type):
    return np.dtype(numpy_type).itemsize


real_type = np.float32
size_type = np.uint32


# setup command line arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("--size", help="the number of data points", type=int, required=True)
parser.add_argument("--dims", help="the number of dimensions per data point", type=int, required=True)
parser.add_argument("--output_file", help="the file to write the generated data points to", type=str, required=True)
parser.add_argument("--num_cluster", help="the number of different clusters", type=int, default=3, required=False)
parser.add_argument("--cluster_std", help="the clusters standard deviation", type=float, default=1.0, required=False)
parser.add_argument("--scale", help="scales the data points to [0, 1]", action="store_true")
parser.add_argument("--binary", help="saves the data in binary format", action="store_true")
parser.add_argument("--debug", help="uses debug data", action="store_true")
args = parser.parse_args()


# generate data points
if args.debug:
    data = np.arange(args.size * args.dims, dtype=real_type)
else:
    data = sklearn.datasets.make_blobs(n_samples=args.size, n_features=args.dims, centers=args.num_cluster, \
                                       cluster_std=args.cluster_std, shuffle=True, random_state=1)[0].astype(real_type)


# scale data to [0, 1] if requested
if args.scale:
    sklearn.preprocessing.minmax_scale(data, feature_range=(0, 1), copy=False)


if args.binary:
    # write data points to file in binary format
    with open(args.output_file, 'wb') as file:
        file.write((args.size).to_bytes(size_in_bytes(size_type), sys.byteorder))
        file.write((args.dims).to_bytes(size_in_bytes(size_type), sys.byteorder))
        file.write(data.tobytes())
else:
    # write data points to file in text format
    with open(args.output_file, 'w', newline='\n') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([args.size])
        writer.writerow([args.dims])
        writer.writerows(data)


# draw data points if dims == 2 || dims == 3
if args.dims == 2:
    plt.scatter(data[:, 0], data[:, 1], s=10)
    plt.show()
elif args.dims == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=10)
    plt.show()