/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the enumeration containing all supported profiling types.
 */

#ifndef SYCL_LSH_PROFILING_TYPES_HPP
#define SYCL_LSH_PROFILING_TYPES_HPP
#pragma once

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // std::ostream and std::istream forward declarations

namespace sycl_lsh {

/**
 * @brief Enum class for the different profiling types.
 */
enum class profiling_types {
    /** no profiling enabled */
    none,
    /** basic profiling for runtimes enabled */
    runtimes,
    /** additional hardware profiling using the hws library enabled (implies runtimes) */
    hws
};

/**
 * @brief Output the @p profiling_type to the given output-stream @p out.
 * @param[in, out] out the output-stream to write the profiling type to
 * @param[in] profiling_type the profiling type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, profiling_types profiling_type);

/**
 * @brief Use the input-stream @p in to initialize the @p profiling_type type.
 * @param[in,out] in input-stream to extract the profiling type from
 * @param[in] profiling_type the profiling type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, profiling_types &profiling_type);

}  // namespace sycl_lsh

template <>
struct fmt::formatter<sycl_lsh::profiling_types> : fmt::ostream_formatter { };

#endif  // SYCL_LSH_PROFILING_TYPES_HPP
