#ifndef DISTRIBUTED_GPU_LSH_USING_SYCL_OPTIONS_HPP
#define DISTRIBUTED_GPU_LSH_USING_SYCL_OPTIONS_HPP

#include <config.hpp>

struct options {
    const integer_t size = 4;
    const integer_t dims = 3;

    const integer_t k = 6;
    const integer_t number_of_hash_tables = 2;
    const integer_t number_of_hash_functions = 3;

    const real_t w = 1.0;
    const integer_t prim = 105613;

    const integer_t local_size = 256;
    const integer_t global_size = ((size + local_size - 1) / local_size) * local_size;
};

#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_OPTIONS_HPP
