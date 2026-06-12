/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements constants used across the code.
 */

#ifndef SYCL_LSH_CONSTANTS_HPP
#define SYCL_LSH_CONSTANTS_HPP

#include <cstdint>      // std::uint32_t, std::uint64_t
#include <type_traits>  // std::is_floating_point_v, std::is_same_v

namespace sycl_lsh {

/// The used floating point type.
#if defined(SYCL_LSH_USE_64BIT_TYPES)
using real_type = double;
#else
using real_type = float;
#endif

/// The used index type.
#if defined(SYCL_LSH_USE_64BIT_TYPES)
using index_type = std::uint64_t;
#else
using index_type = std::uint32_t;
#endif

/// The used hash value type.
#if defined(SYCL_LSH_USE_64BIT_TYPES)
using hash_value_type = std::uint64_t;
#else
using hash_value_type = std::uint32_t;
#endif

/// The used internal blocking size used to speed up the calculations.
/// NOTE: Changing the BLOCKING_SIZE can result in different accuracies since additional points from other hash buckets are considered.
#if defined(SYCL_LSH_BLOCKING_SIZE)
constexpr index_type BLOCKING_SIZE = SYCL_LSH_BLOCKING_SIZE;
#else
constexpr index_type BLOCKING_SIZE = 10;
#endif

// Perform some compile time sanity checks.
static_assert(std::is_floating_point_v<real_type>, "The real_type must be a floating point type!");
static_assert(std::is_same_v<index_type, std::uint32_t> || std::is_same_v<index_type, std::uint64_t>, "The index_type must be a 32bit or 64bit unsigned integer type!");
static_assert(std::is_same_v<hash_value_type, std::uint32_t> || std::is_same_v<hash_value_type, std::uint64_t>, "The hash_value_type must be a 32bit or 64bit unsigned integer type!");
static_assert(BLOCKING_SIZE > 0, "BLOCKING_SIZE must be greater than 0!");

}  // namespace sycl_lsh

#endif  // SYCL_LSH_CONSTANTS_HPP
