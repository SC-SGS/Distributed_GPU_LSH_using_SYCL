# Master Thesis: Distributed k-nearest Neighbors using Locality Sensitive Hashing and SYCL

**Student:** _Marcel Breyer_

**Examiner:** _Prof. Dr. rer. nat. Dirk Pflüger_

**Supervisor:** _M.Sc. Gregor Daiß_

## Description
**TODO**

## Prerequisite
A working [SYCL](https://www.khronos.org/sycl/) installation. Tested with [hipSYCL](https://github.com/illuhad/hipSYCL).

Install via [spack](https://spack.readthedocs.io/en/latest/) 
(also installs all other dependes, like [CMake](https://cmake.org/)):
```bash
$ spack install hipsycl@master build_type=RelWithDebInfo cuda=True
```

To be able to build the documentation [doxygen](https://github.com/doxygen/doxygen) is needed.

## Building
To build the code use:
```bash
$ git@gitlab-sim.informatik.uni-stuttgart.de:breyerml/distributed_gpu_lsh_using_sycl.git
$ cd distributed_gpu_lsh_using_sycl
$ mkdir build && cd build
$ cmake [options] ..
$ make -j $(nprocs)
```

Supported configuration options are:
* `-DCMAKE_BUILD_TYPE=Debug/Release/...` (default: `RelWithDebInfo`)
* `-DENABLE_TESTS=On/Off`: uses the [googletest framework](https://github.com/google/googletest) (automatically installed if this option is set to `On`) to enable the target `test` (default: `Off`)
* `-DENABLE_DOCUMENTATION=On/Off`: enables the target `doc` documentation; requires Doxygen (default: `Off`)
* `-DENABLE_GPU=On/Off`: enables the usage of GPUs (default: `On`)

## Building documentation
After the call to `cmake` use:
```bash
$ make doc
```

## Running test
After a successful `make` (with a previously `cmake` call with option `-DENABLE_TESTS=On`) use:
```bash
$ ctest
```