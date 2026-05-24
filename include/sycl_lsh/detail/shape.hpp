/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Defines a small wrapper struct around a two-dimensional shape.
 */

#ifndef SYCL_LSH_DETAIL_SHAPE_HPP
#define SYCL_LSH_DETAIL_SHAPE_HPP
#pragma once

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <cstddef>     // std::size_t
#include <functional>  // std::hash
#include <iosfwd>      // forward declare std::ostream and std::istream

namespace sycl_lsh::detail {

/**
 * @brief A struct representing a two-dimensional shape.
 * @details Used in the size specification of the matrix class and the device pointers.
 */
struct [[nodiscard]] shape {
    /**
     * @brief Default construct a shape of size 0x0.
     */
    shape() noexcept = default;
    /**
     * @brief Construct a shape of size @p x_p x @p y_p.
     * @details Explicit to prevent conversions from `{ 2, 2 }` to a `sycl_lsh::detail::shape`.
     * @param[in] x_p the shape in x-dimension
     * @param[in] y_p the shape in y-dimension
     */
    explicit shape(std::size_t x_p, std::size_t y_p) noexcept;

    /**
     * @brief Swap the shape dimensions of `*this` with the ones of @p other.
     * @param[in,out] other the other shape
     */
    void swap(shape &other) noexcept;

    /// The shape in the `x` dimension.
    std::size_t x{ 0 };
    /// The shape in the `y` dimension.
    std::size_t y{ 0 };
};

/**
 * @brief Output the shape @p s to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the shape to
 * @param[in] s the shape
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, shape s);

/**
 * @brief Use the input-stream @p in to initialize the shape @p s.
 * @param[in,out] in input-stream to extract the shape from
 * @param[in] s the shape
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, shape &s);

/**
 * @brief Swap the shape dimensions of @p lhs with the ones of @p rhs.
 * @param[in,out] lhs the first shape
 * @param[in,out] rhs the second shape
 */
void swap(shape &lhs, shape &rhs) noexcept;

/**
 * @brief Check whether the shapes @p lhs and @p rhs are equal.
 * @param[in] lhs the first shape
 * @param[in] rhs the second shape
 * @return `true` if `lhs.x == rhs.x` **and** `lhs.y == rhs.y`, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool operator==(shape lhs, shape rhs) noexcept;
/**
 * @brief Check whether the shapes @p lhs and @p rhs are unequal.
 * @param[in] lhs the first shape
 * @param[in] rhs the second shape
 * @return `false` if `lhs.x == rhs.x` **and** `lhs.y == rhs.y`, otherwise `true` (`[[nodiscard]]`)
 */
[[nodiscard]] bool operator!=(shape lhs, shape rhs) noexcept;

}  // namespace sycl_lsh::detail

namespace std {

/**
 * @brief Hashing struct specialization in the `std` namespace for a shape.
 * @details Necessary to be able to use a shape, e.g., in a `std::unordered_map`.
 */
template <>
struct hash<sycl_lsh::detail::shape> {
    /**
     * @brief Overload the function call operator for a default_value.
     * @details Based on Boost's hash_combine for hashing std::pair<T, U>.
     * @param[in] s the shape to hash
     * @return the hash value of @p s
     */
    std::size_t operator()(const sycl_lsh::detail::shape &s) const noexcept {
        // based on boost::hash<std::pair<T, U>>>
        const auto hash_combine = [](std::size_t &seed, const std::size_t val) {
            seed ^= std::hash<std::size_t>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  // NOLINT: taken from boost::hash_combine
        };

        std::size_t seed = 0;
        hash_combine(seed, s.x);
        hash_combine(seed, s.y);
        return seed;
    }
};

}  // namespace std

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<sycl_lsh::detail::shape> : fmt::ostream_formatter { };

/// @endcond

#endif  // SYCL_LSH_DETAIL_SHAPE_HPP
