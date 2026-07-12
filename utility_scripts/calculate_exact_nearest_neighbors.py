#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
# Authors: Marcel Breyer     #
# Copyright (C): 2020-today  #
##############################

import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

def load_dataset(path):
    with open(path, "rb") as f:
        real_type_size = int(np.frombuffer(f.read(np.dtype(index_type).itemsize), dtype=index_type)[0])
        parsing_type_size = int(np.dtype(real_type).itemsize)
        if real_type_size != parsing_type_size:
            raise ValueError(f"The data was stored using a {real_type_size} Byte type but is now read using a {parsing_type_size} Byte type which is not supported!")
        num_data_points = int(np.frombuffer(f.read(np.dtype(index_type).itemsize), dtype=index_type)[0])
        num_dimensions = int(np.frombuffer(f.read(np.dtype(index_type).itemsize), dtype=index_type)[0])
        data = np.frombuffer(f.read(num_data_points * num_dimensions * parsing_type_size), dtype=real_type)
    return data.reshape(num_data_points, num_dimensions)


def save_indices(path, indices):
    size, k = indices.shape
    with open(path, "wb") as f:
        f.write(index_type(np.dtype(index_type).itemsize).tobytes())
        f.write(index_type(size).tobytes())
        f.write(index_type(k).tobytes())
        f.write(indices.astype(index_type).tobytes())


def save_distances(path, distances):
    size, k = distances.shape
    with open(path, "wb") as f:
        f.write(index_type(np.dtype(real_type).itemsize).tobytes())
        f.write(index_type(size).tobytes())
        f.write(index_type(k).tobytes())
        f.write(distances.astype(real_type).tobytes())


if __name__ == "__main__":
    # setup command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="input dataset file")
    parser.add_argument("--knn", type=int, required=True, help="number of nearest neighbors")
    parser.add_argument("--knn_indices_file", type=str, required=True, help="output file for neighbor indices")
    parser.add_argument("--knn_distances_file", type=str, required=True, help="output file for neighbor distances")
    parser.add_argument("--use_64bit_real_type", help="use 64bit real_type instead of 32bit", action="store_true")
    args = parser.parse_args()

    # set the used real_type and index_type
    if args.use_64bit_real_type:
        real_type = np.float64
    else:
        real_type = np.float32
    index_type = np.uint64


    # load the data set in the custom binary format
    points = load_dataset(args.input_file)
    print(f"Loaded {points.shape[0]} points with {points.shape[1]} dims from '{args.input_file}'")

    # calculate the exact k-nearest-neighbors using the brute-force algorithm
    start = time.perf_counter()
    k_nearest_neighbors = NearestNeighbors(n_neighbors=args.knn, algorithm="brute", metric="euclidean")
    k_nearest_neighbors.fit(points)
    nn_distances, nn_ids = k_nearest_neighbors.kneighbors()
    end = time.perf_counter()
    print(f"k-nearest-neighbor total runtime: {end - start:.3f}s")

    # save the nearest-neighbor IDs
    save_indices(args.knn_indices_file, nn_ids)
    print(f"Saved indices to '{args.knn_indices_file}'")

    # save the nearest-neighbors distances
    save_distances(args.knn_distances_file, nn_distances)
    print(f"Saved distances to '{args.knn_distances_file}'")
