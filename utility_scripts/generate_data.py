#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
# Authors: Marcel Breyer     #
# Copyright (C): 2020-today  #
##############################

import argparse
import sklearn.datasets
import sklearn.preprocessing
import numpy as np
import sys
import csv

if __name__ == "__main__":
    # setup command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="the number of data points", type=int, required=True)
    parser.add_argument("--dims", help="the number of dimensions per data point", type=int, required=True)
    parser.add_argument("--output_file", help="the file to write the generated data points to", type=str, required=True)
    parser.add_argument("--num_cluster", help="the number of different clusters", type=int, default=3, required=False)
    parser.add_argument("--cluster_std", help="the clusters standard deviation", type=float, default=1.0, required=False)
    parser.add_argument("--scale", help="scales the data points to [0, 1]", action="store_true")
    parser.add_argument("--binary", help="saves the data in binary format", action="store_true")
    parser.add_argument("--use_64bit_real_type", help="use 64bit real_type instead of 32bit", action="store_true")
    parser.add_argument("--debug", help="uses debug data", action="store_true")
    args = parser.parse_args()

    # set the used real_type and index_type
    if args.use_64bit_real_type:
        real_type = np.float64
    else:
        real_type = np.float32
    index_type = np.uint64

    # generate data points
    if args.debug:
        data = np.arange(args.size * args.dims, dtype=real_type) % args.size
        data = np.reshape(data, (args.size, args.dims))
    else:
        data = sklearn.datasets.make_blobs(n_samples=args.size, n_features=args.dims, centers=args.num_cluster,
                                           cluster_std=args.cluster_std, shuffle=True, random_state=1)[0].astype(real_type)

    # scale data to [0, 1] if requested
    if args.scale:
        sklearn.preprocessing.minmax_scale(data, feature_range=(0, 1), copy=False)

    if args.binary:
        # write data points to file in binary format
        with open(args.output_file, 'wb') as file:
            size_in_bytes = np.dtype(index_type).itemsize
            real_type_size_in_bytes = np.dtype(real_type).itemsize
            file.write(real_type_size_in_bytes.to_bytes(size_in_bytes, sys.byteorder))
            file.write(args.size.to_bytes(size_in_bytes, sys.byteorder))
            file.write(args.dims.to_bytes(size_in_bytes, sys.byteorder))
            file.write(data.tobytes())
    else:
        # write data points to file in text format
        with open(args.output_file, 'w', newline='\n') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([args.size])
            writer.writerow([args.dims])
            writer.writerows(data)
