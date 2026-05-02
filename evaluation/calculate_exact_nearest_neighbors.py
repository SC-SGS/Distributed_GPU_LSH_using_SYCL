#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
# Authors: Marcel Breyer     #
# Copyright (C): 2020-today  #
##############################

import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors

real_type = np.float32
index_type = np.uint32


def load_dataset(path):
    with open(path, "rb") as f:
        size = np.frombuffer(f.read(4), dtype=index_type)[0]
        dims = np.frombuffer(f.read(4), dtype=index_type)[0]
        data = np.frombuffer(f.read(size * dims * 4), dtype=real_type)
    return data.reshape(size, dims)


def save_indices(path, indices):
    size, k = indices.shape
    with open(path, "wb") as f:
        f.write(index_type(size).tobytes())
        f.write(index_type(k).tobytes())
        f.write(indices.astype(index_type).tobytes())


def save_distances(path, distances):
    size, k = distances.shape
    with open(path, "wb") as f:
        f.write(index_type(size).tobytes())
        f.write(index_type(k).tobytes())
        f.write(distances.astype(real_type).tobytes())


if __name__ == "__main__":
    # setup command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="input dataset file")
    parser.add_argument("--knn", type=int, required=True, help="number of nearest neighbors")
    parser.add_argument("--knn_save_file", type=str, required=True, help="output file for neighbor indices")
    parser.add_argument("--knn_dist_save_file", type=str, required=True, help="output file for neighbor distances")
    args = parser.parse_args()

    # load the data set in the custom binary format
    points = load_dataset(args.input_file)
    print(f"Loaded {points.shape[0]} points with {points.shape[1]} dims from '{args.input_file}'")

    # k+1 because results include each point itself
    k_nearest_neighbors = NearestNeighbors(n_neighbors=args.knn + 1, algorithm="brute", metric="euclidean")
    k_nearest_neighbors.fit(points)
    nn_distances, nn_ids = k_nearest_neighbors.kneighbors(points)

    # strip the first column (each point matched to itself)
    nn_distances = nn_distances[:, 1:]
    nn_ids = nn_ids[:, 1:]

    # save the nearest-neighbor IDs
    save_indices(args.knn_save_file, nn_ids)
    print(f"Saved IDs to '{args.knn_save_file}'")

    # save the nearest-neighbors distances
    save_distances(args.knn_dist_save_file, nn_distances)
    print(f"Saved distances to '{args.knn_dist_save_file}'")
