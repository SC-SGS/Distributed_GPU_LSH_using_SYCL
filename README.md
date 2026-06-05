# Performance-Portable Distributed k-Nearest Neighbors using Locality-Sensitive Hashing and SYCL

**Authors:** _Marcel Breyer_, _M.Sc. Gregor Daiß_, _Prof. Dr. rer. nat. Dirk Pflüger_

## Abstract

In the age of data collection, machine learning algorithms have to be able to efficiently cope with vast data sets. This
requires scalable algorithms and efficient implementations that can cope with heterogeneous hardware. We propose a new,
performance-portable implementation of a well-known, robust, and versatile multi-class classification method that
supports multiple Graphics Processing Units (GPUs) from different vendors. It is based on a performance-portable
implementation of the approximate k-nearest neighbors (k-NN) algorithm in SYCL. The k-NN assigns a class to a data point
based on a majority vote of its neighborhood. The naive approach compares a data point x to all other data points in the
training data to identify the k nearest ones. However, this has quadratic runtime and is infeasible for large data sets.
Therefore, approximate variants have been developed. Such an algorithm is the Locality-Sensitive Hashing (LSH)
algorithm, which uses hash tables together with locality-sensitive hash functions to reduce the data points that have to
be examined to compute the k-NN.

To the best of our knowledge, there is no distributed LSH version supporting multiple GPUs from different vendors
available so far despite the fact that k-NNs are frequently employed. Therefore, we have developed the library. It
provides the first hardware-independent, yet efficient and distributed implementation of the LSH algorithm that is
suited for modern supercomputers. The implementation uses C++17 together with SYCL 1.2.1, which is an abstraction layer
for OpenCL that allows targeting different hardware with a single implementation. To support large data sets, we utilize
multiple GPUs using the Message Passing Interface (MPI) to enable the usage of both shared and distributed memory
systems.

We have tested different parameter combinations for two locality-sensitive hash function implementations, which we
compare. Our results show that our library can easily scale on multiple GPUs using both hash function types, achieving a
nearly optimal parallel speedup of up to 7.6 on 8 GPUs. Furthermore, we demonstrate that the library supports different
SYCL implementations—ComputeCpp, hipSYCL, and DPC++—to target different hardware architectures without significant
performance differences.

### Citing this work

```
@inbook{10.1145/3456669.3456692,
	author = {Breyer, Marcel and Dai\ss{}, Gregor and Pfl\"{u}ger, Dirk},
	title = {Performance-Portable Distributed k-Nearest Neighbors Using Locality-Sensitive Hashing and SYCL},
	year = {2021},
	isbn = {9781450390330},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3456669.3456692},
	booktitle = {International Workshop on OpenCL},
	articleno = {4},
	numpages = {12}
}
```

## Pre-requisite

* A working [SYCL](https://www.khronos.org/sycl/) installation, i.e., [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp), or [icpx](https://github.com/intel/llvm).
* An MPI implementation, i.e., [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/).
* The [{fmt}](https://github.com/fmtlib/fmt) formatting library (automatically build if not found).
* The [cxxopts](https://github.com/jarro2783/cxxopts) command line argument parsing library (automatically build if not found).
* [CMake](https://cmake.org/) 3.25 or newer.
* [Doxygen](https://github.com/doxygen/doxygen) (optional) to build the documentation.

## Before building

1. Install one of the two mentioned SYCL implementations.
2. Export the variable `SYCL_LSH_TARGET` given your target choice. Examples:
    - CPUs: `export SYCL_LSH_TARGET=spir64_x86_64`
    - NVIDIA GPUs: `export SYCL_LSH_TARGET=nvidia_gpu_sm_80`
    - AMD GPUs: `export SYCL_LSH_TARGET=amd_gpu_gfx90a`
    - Intel GPUs: `export SYCL_LSH_TARGET=intel_gpu_bmg_g21`

The value can be overwritten by using `-DSYCL_LSH_TARGET=XXX` when invoking CMake.

## Building the program

To build the code use:

```bash
git clone git@github.com:SC-SGS/Distributed_GPU_LSH_using_SYCL.git
cd Distributed_GPU_LSH_using_SYCL
cmake --preset [preset] [options] .
cmake --build --preset [preset]
```

Where `preset` is one of `acpp`, or `icpx`.

Provided configuration options are:

| option                                | default value | description                                                                                                                                         |
|---------------------------------------|:-------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `SYCL_LSH_TARGET_ARCH`                |               | Specify the SYCL target to compile for. Must follow the notation in aboves examples.                                                                | 
| `SYCL_LSH_CPU_VECTORIZATION_TARGET`   |               | If the SYCL target is a CPU, this option **must** be equal to the target vectorization standard. Must be one of: `avx512`, `avx2`, `avx`, or `sse`. | 
| `SYCL_LSH_IMPLEMENTATION`             |    `icpx`     | The used SYCL implementation. Must be one of: `acpp`, or `icpx`.                                                                                    | 
| `SYCL_LSH_ENABLE_LTO`                 |     `OFF`     | Enable link time optimizations.                                                                                                                     | 
| `SYCL_LSH_TIMER`                      |  `BLOCKING`   | Specify which timer functionality should be used. Must be one of: `NON_BLOCKING`, or `BLOCKING`.                                                    |
| `SYCL_LSH_HARDWARE_SAMPLING_INTERVAL` |     `100`     | Specify the hardware sampling interval in ms.                                                                                                       |
| `SYCL_LSH_ENABLE_ASSERTS`             |     `OFF`     | Enables assertion macros for sanity checks.                                                                                                         |
| `SYCL_LSH_RANDOM_NUMBERS_DEBUG`       |     `OFF`     | If `ON`, do not seed the random number generators to enable better reproducability (can be used for debugging).                                     |
| `SYCL_LSH_USE_64BIT_TYPES`            |     `OFF`     | If `ON`, internally uses 64bit tyües instead of 32bit types.                                                                                        |
| `SYCL_LSH_ENABLE_DOCUMENTATION`       |     `OFF`     | Enables the documentation target (requires doxygen).                                                                                                |

## Building the documentation

After the call to `cmake --preset [preset] -DENABLE_DOCUMENTATION=ON .` use:

```bash
cmake --build --preset [preset] --target doc
```

## Running the program

After a successful `make` an executable file named `./prog` is available:

```bash
./prog --help
k-nearest-neighbors using Locality Sensitive Hashing and SYCL
Usage:
  build/icpx/prog [OPTION...] input_file k

  -h, --help                    print this helper message
      --file_parser arg         the type of the file parser: 
                                        0: binary
                                        1: arff (default: binary)
      --profiling_type arg      the profiling capabilities: 
                                        0: none
                                        1: runtimes
                                        2: hws (default: none)
      --profiling_file arg      the output file to write the profiling results to (YAML format)
      --indices_save_file arg   the file to which the calculated nearest-neighbors should be saved to
      --distances_save_file arg
                                the file to which the calculated nearest-neighbors distances should be saved to
      --indices_ground_truth_file arg
                                the file containing the correct nearest-neighbors for calculating the resulting recall
      --distances_ground_truth_file arg
                                the file containing the correct nearest-neighbors distances for calculating the resulting recall
      --hash_function arg       the type of the hash functions: 
                                        0: random-projections
                                        1: entropy-based
                                        2: mixed (default: random_projections)
      --hash_pool_size arg      the number of hash functions in the hash pool (default: 32)
      --num_hash_functions arg  the number of hash functions per hash table (default: 12)
      --num_hash_tables arg     the number of used hash tables (default: 8)
      --hash_table_size arg     the size of each hash table (default: 105613)
  -w arg                        the segment size for the random projections hash functions (default: 1.0)
      --num_cut_off_points arg  the number of cut-off points for the entropy-based hash functions (default: 6)
      --file input_file         the input data file
      --knn knn                 the number of nearest-neighbors to calculate

```
