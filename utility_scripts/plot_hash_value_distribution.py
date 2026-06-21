#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
# Authors: Marcel Breyer     #
# Copyright (C): 2020-today  #
##############################

import argparse
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # setup command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="input distribution file")
    parser.add_argument("--output_file", type=str, required=False, help="output plot file")
    args = parser.parse_args()

    # load the data
    distributions = genfromtxt(args.input_file, delimiter=',')

    for dist in distributions:
        data = dist[1:]
        plt.plot(np.arange(len(data)), data, label=f"hash table {dist[0]}")
        plt.yscale("log")

    if args.output_file is not None:
        plt.savefig(args.output_file, dpi=150, bbox_inches="tight")
    else:
        plt.show()
