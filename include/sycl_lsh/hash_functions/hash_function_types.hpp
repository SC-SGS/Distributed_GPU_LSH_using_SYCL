/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the enumeration containing all support hash function types.
 */

#ifndef SYCL_LSH_HASH_FUNCTIONS_HASH_FUNCTION_TYPES_HPP
#define SYCL_LSH_HASH_FUNCTIONS_HASH_FUNCTION_TYPES_HPP
#pragma once

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // std::ostream and std::istream forward declarations

namespace sycl_lsh {

/**
 * @brief Enum class for the different hash function types.
 */
enum class hash_function_type {
    /** random projections hash functions */
    random_projections,
    /** entropy based hash functions */
    entropy_based,
    /** mixed hash functions (random projections + entropy-based as hash combine) */
    mixed_hash_functions
};

/**
 * @brief Output the @p hash_function to the given output-stream @p out.
 * @param[in, out] out the output-stream to write the hash function type to
 * @param[in] hash_function the hash function type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, hash_function_type hash_function);

/**
 * @brief Use the input-stream @p in to initialize the @p hash_function type.
 * @param[in,out] in input-stream to extract the hash function type from
 * @param[in] hash_function the hash function type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, hash_function_type &hash_function);

}  // namespace sycl_lsh

template <>
struct fmt::formatter<sycl_lsh::hash_function_type> : fmt::ostream_formatter { };

#endif  // SYCL_LSH_HASH_FUNCTIONS_HASH_FUNCTION_TYPES_HPP
