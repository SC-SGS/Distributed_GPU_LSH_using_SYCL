# Performance-Portable Distributed k-Nearest Neighbors using Locality-Sensitive Hashing and SYCL

**Authors:** _Marcel Breyer_, _M.Sc. Gregor Daiß_, _Prof. Dr. rer. nat. Dirk Pflüger_


## Abstract

In the age of data collection, machine learning algorithms have to be able to efficiently cope with vast data sets. This requires scalable algorithms and efficient implementations that can cope with heterogeneous hardware. We propose a new, performance-portable implementation of a well-known, robust, and versatile multi-class classification method that supports multiple Graphics Processing Units (GPUs) from different vendors. It is based on a performance-portable implementation of the approximate k-nearest neighbors (k-NN) algorithm in SYCL. The k-NN assigns a class to a data point based on a majority vote of its neighborhood. The naive approach compares a data point x to all other data points in the training data to identify the k nearest ones. However, this has quadratic runtime and is infeasible for large data sets. Therefore, approximate variants have been developed. Such an algorithm is the Locality-Sensitive Hashing (LSH) algorithm, which uses hash tables together with locality-sensitive hash functions to reduce the data points that have to be examined to compute the k-NN.

To the best of our knowledge, there is no distributed LSH version supporting multiple GPUs from different vendors available so far despite the fact that k-NNs are frequently employed. Therefore, we have developed the library. It provides the first hardware-independent, yet efficient and distributed implementation of the LSH algorithm that is suited for modern supercomputers. The implementation uses C++17 together with SYCL 1.2.1, which is an abstraction layer for OpenCL that allows targeting different hardware with a single implementation. To support large data sets, we utilize multiple GPUs using the Message Passing Interface (MPI) to enable the usage of both shared and distributed memory systems.

We have tested different parameter combinations for two locality-sensitive hash function implementations, which we compare. Our results show that our library can easily scale on multiple GPUs using both hash function types, achieving a nearly optimal parallel speedup of up to 7.6 on 8 GPUs. Furthermore, we demonstrate that the library supports different SYCL implementations—ComputeCpp, hipSYCL, and DPC++—to target different hardware architectures without significant performance differences.

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
* A working [SYCL](https://www.khronos.org/sycl/) installation, i.e., [hipSYCL](https://github.com/illuhad/hipSYCL), 
  [ComputeCpp](https://developer.codeplay.com/products/computecpp/ce/guides), or [DPC++](https://github.com/intel/llvm).
* A MPI implementation, i.e., [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/).
* The [{fmt}](https://github.com/fmtlib/fmt) formatting library (version 7.1.0 required to be able to format a `std::chrono::time_point`).
* [CMake](https://cmake.org/) 3.20 or newer (in order to use presets)
* [doxygen](https://github.com/doxygen/doxygen) (optional) to build the documentation.


## Before building

1. Install one of the three mentioned SYCL implementations.
2. Based on the chosen implementation set the environment variables `HIPSYCL_INSTALL_DIR`, `DPCPP_INSTALL_DIR` or `COMPUTECPP_INSTALL_DIR` 
   and `COMPUTECPP_SDK_INSTALL_DIR` to the respective directories.
3. Export the variable `SYCL_LSH_TARGET` given your target choice (based on hipSYCL's notation):
    - CPUs: `export SYCL_LSH_TARGET=omp`
    - NVIDIA GPUs: `export SYCL_LSH_TARGET=cuda:sm_XX`
    - AMD GPUs: `export SYCL_LSH_TARGET=hip:gfxXXX`
    - Intel GPUs: `export SYCL_LSH_TARGET=spirv`

   The value can be overwritten by using `-DSYCL_LSH_TARGET=XXX` when invoking CMake.


## Building the program

To build the code use:
```bash
$ git clone git@github.com:SC-SGS/Distributed_GPU_LSH_using_SYCL.git
$ cd Distributed_GPU_LSH_using_SYCL
$ cmake --preset [preset] [options] .
$ cmake --build --preset [preset]
```

Where `preset` is one of `hipsycl`, `computecpp`, or `dpcpp`.

Provided configuration options are:

| option                                 | default value | description                                                                                                                                                                        |
|----------------------------------------|:-------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `SYCL_LSH_TARGET`                      |               | Specify the SYCL target to compile for. Must be one of: `omp` (CPUs), `cuda:sm_XX` (NVIDIA GPUs), `hip:gfxXXX` (AMD GPUs), or `spirv` (Intel GPUs). | 
| `SYCL_LSH_TIMER`                       | `BLOCKING`    | Specify which timer functionality should be used. Must be one of: `NONE`, `NON_BLOCKING`, or `BLOCKING`.                                            |
| `SYCL_LSH_BENCHMARK`                   |               | If defined enables benchmarking by logging the elapsed times in a machine readable way to a file. Must be a valid file name.                        |
| `SYCL_LSH_ENABLE_DEBUG`                | `OFF`         | Enables the debugging macros.                                                                                                                       |
| `SYCL_LSH_ENABLE_DOCUMENTATION`        | `OFF`         | Enables the documentation target (requires doxygen).                                                                                                |


## Building the documentation

After the call to `cmake --preset [preset] -DENABLE_DOCUMENTATION=ON .` use:
```bash
$ cmake --build --preset [preset] --target doc
```


## Running the program
After a successful `make` an executable file named `./prog` is available:
```bash
$ ./prog --help
Usage: ./prog --data_file "path_to_data_file" --k "number_of_knn_to_search" [options]
options:
   --data_file                path to the data file (required)
   --evaluate_knn_dist_file   read the correct nearest-neighbor distances for calculating the error ratio 
   --evaluate_knn_file        read the correct nearest-neighbors for calculating the resulting recall 
   --file_parser              type of the file parser 
   --hash_pool_size           number of hash functions in the hash pool 
   --hash_table_size          size of each hash table 
   --help                     help screen 
   --k                        the number of nearest-neighbors to search for (required)
   --knn_dist_save_file       save the calculated nearest-neighbor distances to path 
   --knn_save_file            save the calculated nearest-neighbors to path 
   --num_cut_off_points       number of cut-off points for the entropy-based hash functions 
   --num_hash_functions       number of hash functions per hash table 
   --num_hash_tables          number of hash tables to create 
   --options_file             path to options file 
   --options_save_file        save the currently used options to the given path 
   --w                        segment size for the random projections hash functions     
```
