/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements an enum class for the different file parser types.
 */

#ifndef SYCL_LSH_MPI_FILE_PARSER_TYPES_HPP
#define SYCL_LSH_MPI_FILE_PARSER_TYPES_HPP
#pragma once

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // std::ostream and std::istream forward declarations

namespace sycl_lsh::mpi {

/**
 * @brief Enum class for all available file parser types.
 */
enum class file_parser_type {
    /** Parser used for the (custom) binary files. */
    binary,
    /** Parser used for general ARFF files. */
    arff
};

/**
 * @brief Output the @p parser to the given output-stream @p out.
 * @param[in, out] out the output-stream to write the parser type to
 * @param[in] parser the parser type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, file_parser_type parser);

/**
 * @brief Use the input-stream @p in to initialize the @p parser type.
 * @param[in,out] in input-stream to extract the parser type from
 * @param[in] parser the parser type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, file_parser_type &parser);

}  // namespace sycl_lsh::mpi

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<sycl_lsh::mpi::file_parser_type> : fmt::ostream_formatter { };

/// @endcond

#endif  // SYCL_LSH_MPI_FILE_PARSER_TYPES_HPP
