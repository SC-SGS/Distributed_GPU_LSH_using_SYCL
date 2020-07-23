# @author Marcel Breyer
# @date 2020-07-23
# @brief Python3 script for converting a .arff file in text format to a file in binary format.


import argparse
from scipy.io import arff
import numpy as np
import sys

def size_in_bytes(numpy_type):
    return np.dtype(numpy_type).itemsize


real_type = np.float32
# real_type = np.uint32
size_type = np.uint32


# setup command line arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="the '.arff# file to convert to binary", type=str, required=True)
parser.add_argument("--output_file", help="the file to write the binary representation to", type=str, required=True)
args = parser.parse_args()

if not args.input_file.endswith('.arff'):
    raise ValueError("'{}' is not an .arff file!".format(args.input_file))
if args.output_file.endswith('.arff'):
    raise ValueError("The output file ('{}') should NOT have an '.arff' extension!".format(args.output_file))

# read .arff file and convert it to the custom binary format
data = arff.loadarff(args.input_file)[0]
data = np.array(data.tolist(), dtype=real_type)
# data = data[..., :-1]

# write data points to file in binary format
with open(args.output_file, 'wb') as file:
    file.write((data.shape[0]).to_bytes(size_in_bytes(size_type), sys.byteorder))
    file.write((data.shape[1]).to_bytes(size_in_bytes(size_type), sys.byteorder))
    file.write(data.tobytes())
