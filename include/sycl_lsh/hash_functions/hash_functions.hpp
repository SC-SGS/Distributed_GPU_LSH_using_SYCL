/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-30
 *
 * @brief Implements the factory functions for the hash functions classes.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTIONS_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTIONS_HPP

#include <ostream>

namespace sycl_lsh {

    struct hash_functions {

        static struct RandomProjection{} random_projection;
        static struct EntropyBased{} entropy_based;

    };

    std::ostream& operator<<(std::ostream& out, hash_functions::EntropyBased);
    std::ostream& operator<<(std::ostream& out, hash_functions::RandomProjection);

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTIONS_HPP
