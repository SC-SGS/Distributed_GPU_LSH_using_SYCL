# @author Marcel Breyer
# @date 2020-06-03
# @brief Python3 script for generating data sets.


import argparse
import sklearn.datasets
import sklearn.preprocessing
import csv
import matplotlib.pyplot as plt


# setup command line arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("--size", help="the number of data points", type=int, required=True)
parser.add_argument("--dims", help="the number of dimensions per data point", type=int, required=True)
parser.add_argument("--output_file", help="the file to write the generated data points to", type=str, required=True)
parser.add_argument("--num_cluster", help="the number of different clusters", type=int, default=3, required=False)
parser.add_argument("--cluster_std", help="the clusters standard deviation", type=float, default=1.0, required=False)
parser.add_argument("--scale", help="scales the data points to [0, 1]", action="store_true")
args = parser.parse_args()


# generate data points
X, _ = sklearn.datasets.make_blobs(n_samples=args.size, n_features=args.dims, centers=args.num_cluster,\
                                   cluster_std=args.cluster_std, shuffle=True, random_state=1)


# scale data to [0, 1] if requested
if args.scale:
    sklearn.preprocessing.minmax_scale(X, feature_range=(0, 1), copy=False)


# write data points to file
with open(args.output_file, 'w', newline='\n') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(X)


# print data points if dims == 2
if args.dims == 2:
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.show()
