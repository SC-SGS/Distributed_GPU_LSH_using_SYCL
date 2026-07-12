#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
# Authors: Marcel Breyer     #
# Copyright (C): 2020-today  #
##############################

import argparse
from sklearn.datasets import load_svmlight_file
import numpy as np
import sys

if __name__ == "__main__":
    # setup command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="the libsvm file to convert to binary", type=str, required=True)
    parser.add_argument("--output_file", help="the file to write the binary representation to", type=str, required=True)
    parser.add_argument("--use_64bit_real_type", help="use 64bit real_type instead of 32bit", action="store_true")
    args = parser.parse_args()

    if args.output_file.endswith('.libsvm'):
        raise ValueError(f"The output file ('{args.output_file}') should NOT have an '.libsvm' extension!")

    # set the used real_type and index_type
    if args.use_64bit_real_type:
        real_type = np.float64
    else:
        real_type = np.float32
    index_type = np.uint64

    # read libsvm file and convert it to the custom binary format
    data = load_svmlight_file(args.input_file)[0]
    data = np.array(data.todense(), dtype=real_type)

    # write data points to file in binary format
    with open(args.output_file, 'wb') as file:
        size_in_bytes = np.dtype(index_type).itemsize
        real_type_size_in_bytes = np.dtype(real_type).itemsize
        file.write(real_type_size_in_bytes.to_bytes(size_in_bytes, sys.byteorder))
        file.write((data.shape[0]).to_bytes(size_in_bytes, sys.byteorder))
        file.write((data.shape[1]).to_bytes(size_in_bytes, sys.byteorder))
        file.write(data.tobytes())
