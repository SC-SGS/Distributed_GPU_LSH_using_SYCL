# Master Thesis: Distributed k-nearest Neighbors using Locality Sensitive Hashing and SYCL

**Student:** _Marcel Breyer_

**Examiner:** _Prof. Dr. rer. nat. Dirk Pflüger_

**Supervisor:** _M.Sc. Gregor Daiß_


## Description
The computation of k-Nearest Neighbors (KNN) graphs is an integral part of numerous data mining tasks and scientific 
computing applications. For large amounts of data, as commonly seen in those fields, the data size alone becomes too 
big to fit on a single machine. Furthermore, the sub-quadratic runtime required for the computation of the graph 
naturally increases with the size of the data. Hence, the distributed computation of the KNN graphs using 
locality-sensitive hashing (LSH) becomes an intriguing prospect. While there already are some distributed approaches 
for the implementation of this algorithm, those are generally relying on MapReduce or similar frameworks. For 
High-Performance-Computing a different approach is preferable. The implementation should work on modern accelerators 
and distribute the workload over multiple compute nodes using a framework suitable to be run on supercomputers. This 
work aims to create just such an LSH implementation, enabling the distributed computation of the KNN, using data sets 
that are located over multiple compute nodes using MPI (or HPX as a modern alternative).


## Pre-requisite
* A working [SYCL](https://www.khronos.org/sycl/) installation. For example [hipSYCL](https://github.com/illuhad/hipSYCL), 
  [ComputeCpp](https://developer.codeplay.com/products/computecpp/ce/guides) or [oneAPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html).
* A MPI implementation. For example [OpenMPI](https://www.open-mpi.org/).
* The [{fmt}](https://github.com/fmtlib/fmt) formatting library (version 7.1.0 required to be able to format a `std::chrono::time_point`).
* [doxygen](https://github.com/doxygen/doxygen) (optional) to build the documentation.


## Using hipSYCL as SYCL implementation

1. Install hipSYCL (https://github.com/illuhad/hipSYCL), e.g. using [spack](https://github.com/spack/spack)
2. Export the environment variable `hipSYCL_DIR` to the root directory of the hipSYCL installation.
3. Export the environment variable `hipSYCL_GPU_ARCH` to the used GPU architecture value (e.g. `sm_75`).


## Using ComputeCpp as SYCL implementation

1. Be sure all pre-requisites are installed: https://developer.codeplay.com/products/computecpp/ce/guides.
2. Download and install ComputeCpp: https://developer.codeplay.com/products/computecpp/ce/download.
3. Clone the ComputeCpp SDK github repository: https://github.com/codeplaysoftware/computecpp-sdk.
4. Export the environment variable `ComputeCpp_DIR` to the root directory of the ComputeCpp installation.
5. Export the environment variable `ComputeCpp_SDK_DIR` to the root directory of the ComputeCpp SDK installation.


## Using oneAPI as SYCL implementation

1. Easiest way: login to Intel's devcloud: https://software.intel.com/content/www/us/en/develop/tools/devcloud.html
2. Install a GCC version capable of C++17
3. Export the environment variable `DPCPP_GCC_TOOLCHAIN` to the root directory of the GCC installation.


## Building the program
To build the code use:
```bash
$ git clone git@github.com:SC-SGS/Distributed_GPU_LSH_using_SYCL.git
$ cd Distributed_GPU_LSH_using_SYCL
$ mkdir build && cd build
$ cmake [options] ..
$ make -j $(nprocs)
```

Provided configuration options are:

| option                                 | default value | description                                                                                                                                                                        |
|----------------------------------------|:-------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SYCL_LSH_IMPLEMENTATION`              | `hipSYCL`     | Specify the used SYCL implementation. Must be one of: `hipSYCL`, `ComputeCpp` or `oneAPI` (in case of `oneAPI`: the env variable `DPCPP_GCC_TOOLCHAIN` must be set to a GCC >= 8). |
| `SYCL_LSH_TARGET`                      | `NVIDIA`      | Specify the SYCL target to compile for. Must be one of: `CPU`, `NVIDIA`, `AMD` or `INTEL`.                                                                                                  | 
| `SYCL_LSH_TIMER`                       | `BLOCKING`    | Specify which timer functionality should be used. Must be one of: `NONE`, `NON_BLOCKING` or `BLOCKING`.                                                                            |
| `SYCL_LSH_BENCHMARK`                   |               | If defined enables benchmarking by logging the elapsed times in a machine readable way to a file. Must be a valid file name.                                                       |
| `SYCL_LSH_ENABLE_DEBUG`                | `OFF`         | Enables the debugging macros.                                                                                                                                                      |
| `SYCL_LSH_ENABLE_DOCUMENTATION`        | `OFF`         | Enables the documentation `make` target (requires doxygen).                                                                                                                        |
| `SYCL_LSH_FMT_HEADER_ONLY`             | `OFF`         | Enables `{fmt}` lib's header only mode, otherwise tries to link against it.                                                                                                        |
| `SYCL_LSH_USE_EXPERIMENTAL_FILESYSTEM` | `OFF`         | Enables the `<experimental/filesystem>` header instead of the C++17 `<filesystem>` header.                                                                                         |


## Building the documentation
After the call to `cmake -DENABLE_DOCUMENTATION=ON ..` use:
```bash
$ make doc
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
