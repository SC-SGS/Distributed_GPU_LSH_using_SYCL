/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the @ref sycl_lsh::detail::hashing::hash_combine() function for different hash value type sizes.
 */

#ifndef SYCL_LSH_DETAIL_HASHING_HASH_COMBINE_HPP
#define SYCL_LSH_DETAIL_HASHING_HASH_COMBINE_HPP
#pragma once

#include <cstdint>  // std::uint32_t, std::uint64_t

namespace sycl_lsh::detail::hashing {

/**
 * @brief Combine two std::uint32_t hash values into a new one. Base on boost::hash_combine.
 * @param[in] seed the seed
 * @param[in] val the hash value
 * @return the combined hash value (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::uint32_t hash_combine(const std::uint32_t seed, const std::uint32_t val) noexcept {
    std::uint32_t x = seed + 0x9e3779b9u + val;
    // start the mixing
    x ^= x >> 16u;
    x *= 0x21f0aaadu;
    x ^= x >> 15u;
    x *= 0x735a2d97u;
    x ^= x >> 15u;
    return x;
}

/**
 * @brief Combine two std::uint64_t hash values into a new one. Base on boost::hash_combine.
 * @param[in] seed the seed
 * @param[in] val the hash value
 * @return the combined hash value (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::uint64_t hash_combine(const std::uint64_t seed, const std::uint64_t val) noexcept {
    std::uint64_t x = seed + 0x9e3779b97f4a7c15llu + val;
    // start the mixing
    x ^= x >> 32llu;
    x *= 0xe9846af9b1a615dllu;
    x ^= x >> 32llu;
    x *= 0xe9846af9b1a615dllu;
    x ^= x >> 28llu;
    return x;
}

}  // namespace sycl_lsh::detail::hashing

#endif  // SYCL_LSH_DETAIL_HASHING_HASH_COMBINE_HPP
