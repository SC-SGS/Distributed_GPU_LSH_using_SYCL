# Master Thesis: Distributed k-nearest Neighbors using Locality Sensitive Hashing and SYCL

**Student:** _Marcel Breyer_

**Examiner:** _Prof. Dr. rer. nat. Dirk Pflüger_

**Supervisor:** _M.Sc. Gregor Daiß_

## Description
**TODO**

## Prerequisite
* A working [SYCL](https://www.khronos.org/sycl/) installation. For example [hipSYCL](https://github.com/illuhad/hipSYCL) 
  or [ComputeCpp](https://developer.codeplay.com/products/computecpp/ce/guides) (currently not working).
* A MPI implementation. For example [OpenMPI](https://www.open-mpi.org/).
* [doxygen](https://github.com/doxygen/doxygen) (optional) to build the documentation.

## Building the program
To build the code use:
```bash
$ git clone git@gitlab-sim.informatik.uni-stuttgart.de:breyerml/distributed_gpu_lsh_using_sycl.git
$ cd distributed_gpu_lsh_using_sycl
$ mkdir build && cd build
$ cmake [options] ..
$ make -j $(nprocs)
```

Provided configuration options are:
| option                 | default value | description                                                                                             |
| ---------------------- | :-----------: | ------------------------------------------------------------------------------------------------------- |
| `SYCL_IMPLEMENTATION`  | `hipSYCL`     | Specify the used SYCL implementation. Must be one of: `hipSYCL` or `ComputeCpp`.                        |
| `SYCL_TARGET`          | `NVIDIA`      | Specify the SYCL target to compile for. Must be one of: `CPU`, `NVIDIA` or `AMD`.                       | 
| `TIMER`                | `DETAILED`    | Specify which timer functionality should be used. Must be one of: `NONE`, `REDUCED_FILE` or `DETAILED`. |
| `ENABLE_DOCUMENTATION` | `OFF`         | Enables the documentation `make` target (requires doxygen)                       .                      |


## Building the documentation
After the call to `cmake -DENABLE_DOCUMENTATION=ON ..` use:
```bash
$ make doc
```

## Running the program
After a successful `make` an executable file named `./prog` is available:
```bash
$ ./prog --help
Usage: ./prog --data "path-to-data_set" --k "number-of-knn" [options]
options:
   --data                path to the data file (required)
   --evaluate_knn        read the correct nearest-neighbors and evaluate computed nearest-neighbors
   --hash_pool_size      number of hash functions in the hash pool
   --hash_table_size     size of each hash table (must be a prime)
   --help                help screen
   --k                   the number of nearest-neighbors to search for (required)
   --num_cut_off_points  number of cut-off points for the entropy-based hash functions
   --num_hash_functions  number of hash functions per hash table
   --num_hash_tables     number of hash tables to create
   --options             path to options file
   --save_knn            save the calculate nearest-neighbors to path
   --save_options        save the currently used options to path
   --w                   constant used in the hash functions 
```
