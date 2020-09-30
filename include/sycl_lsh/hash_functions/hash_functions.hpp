/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-30
 *
 * @brief Implements the factory functions for the hash functions classes.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTIONS_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTIONS_HPP

#include <fmt/ostream.h>

#include <ostream>

namespace sycl_lsh {

    /**
     * @brief Enum class for the different hash function types.
     */
    enum class hash_functions_type {
        /** random projections hash functions */
        random_projections,
        /** entropy based hash functions */
        entropy_based
    };

    /**
     * @brief Print the @p type of the hash functions to the output stream @p out.
     * @param[in,out] out the output stream
     * @param[in] type the hash functions type
     * @return the output stream
     */
    std::ostream& operator<<(std::ostream& out, const hash_functions_type type);
    
}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTIONS_HPP
