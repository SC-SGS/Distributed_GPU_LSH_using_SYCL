/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-01
 *
 * @brief Implements an enum class to determine the memory layout type: Array of Structs or Struct of Arrays.
 */

#ifndef SYCL_LSH_INCLUDE_MEMORY_LAYOUT_HPP
#define SYCL_LSH_INCLUDE_MEMORY_LAYOUT_HPP

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>       // std::ostream and std::istream forward declarations
#include <string_view>  // std::string_view

namespace sycl_lsh {

/**
 * @brief Enum class for all available layout types.
 */
enum class memory_layout {
    /** Array-of-Structs (AoS) */
    aos,
    /** Structs-of-Arrays (SoA) */
    soa
};

/**
 * @brief Output the @p layout to the given output-stream @p out.
 * @param[in, out] out the output-stream to write the layout type to
 * @param[in] layout the memory layout type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, memory_layout layout);

/**
 * @brief Use the input-stream @p in to initialize the @p layout type.
 * @param[in,out] in input-stream to extract the layout type from
 * @param[in] layout the memory layout type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, memory_layout &layout);

/**
 * @brief In contrast to operator>> return the full name of the provided @p layout type.
 * @param[in] layout the layout type
 * @return the full name of the layout type (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view layout_type_to_full_string(memory_layout layout);

}  // namespace sycl_lsh

template <>
struct fmt::formatter<sycl_lsh::memory_layout> : fmt::ostream_formatter {};

#endif  // SYCL_LSH_INCLUDE_MEMORY_LAYOUT_HPP
