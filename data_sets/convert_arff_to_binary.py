#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
# Authors: Marcel Breyer     #
# Copyright (C): 2020-today  #
##############################

import argparse
from scipy.io import arff
import numpy as np
import sys

# setup command line arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="the '.arff' file to convert to binary", type=str, required=True)
parser.add_argument("--output_file", help="the file to write the binary representation to", type=str, required=True)
parser.add_argument("--use_64bit_types", help="use 64bit types", action="store_true")
args = parser.parse_args()

if not args.input_file.endswith('.arff'):
    raise ValueError(f"'{args.input_file}' is not an .arff file!")
if args.output_file.endswith('.arff'):
    raise ValueError(f"The output file ('{args.output_file}') should NOT have an '.arff' extension!")

# set the used real_type and index_type
if args.use_64bit_types:
    real_type = np.float64
    index_type = np.uint64
else:
    real_type = np.float32
    index_type = np.uint32

# read .arff file and convert it to the custom binary format
data = arff.loadarff(args.input_file)[0]
data = np.array(data.tolist(), dtype=real_type)

# write data points to file in binary format
with open(args.output_file, 'wb') as file:
    size_in_bytes = np.dtype(index_type).itemsize
    file.write((data.shape[0]).to_bytes(size_in_bytes, sys.byteorder))
    file.write((data.shape[1]).to_bytes(size_in_bytes, sys.byteorder))
    file.write(data.tobytes())
