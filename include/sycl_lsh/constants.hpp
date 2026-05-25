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
#include <type_traits>  // std::is_floating_point_v, std::us_unsigned_v, std::is_same_v

namespace sycl_lsh {

/// The used floating point type.
using real_type = float;

/// The used index type.
using index_type = std::uint32_t;

/// The used hash value type.
using hash_value_type = std::uint32_t;

/// The used internal blocking size used to speed up the calculations.
constexpr index_type BLOCKING_SIZE = 10;

// Perform some compile time sanity checks.
static_assert(std::is_floating_point_v<real_type>, "The real_type must be a floating point type!");
static_assert(std::is_unsigned_v<index_type>, "The index_type must be an integral type!");
static_assert(std::is_same_v<hash_value_type, std::uint32_t> || std::is_same_v<hash_value_type, std::uint64_t>, "The hash_value_type must be a 32bit or 64bit unsigned integer type!");
static_assert(BLOCKING_SIZE > 0, "BLOCKING_SIZE must be greater than 0!");

}  // namespace sycl_lsh

#endif  // SYCL_LSH_CONSTANTS_HPP
