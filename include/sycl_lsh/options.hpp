/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements a @ref sycl_lsh::options class for managing hyperparameters.
 */

#ifndef SYCL_LSH_OPTIONS_HPP
#define SYCL_LSH_OPTIONS_HPP
#pragma once

#include "sycl_lsh/constants.hpp"                          // sycl_lsh::real_type, sycl_lsh::index_type, sycl_lsh::hash_value_type
#include "sycl_lsh/hash_function_types.hpp"                // sycl_lsh::hash_function_type
#include "sycl_lsh/mpi/communicator.hpp"                   // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/logging.hpp"                 // sycl_lsh::mpi::detail::log
#include "sycl_lsh/mpi/file_parser/file_parser_types.hpp"  // sycl_lsh::mpi::file_parser

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter
#include "igor/igor.hpp"  // IGOR_MAKE_NAMED_ARGUMENT, igor::parser, igor::has_unnamed_arguments, igor::has_other_than

#include <iosfwd>    // std::ostream  forward declarations
#include <optional>  // std::optional
#include <string>    // std::string

namespace sycl_lsh {

/// @cond Doxygen_suppress

// create named arguments
IGOR_MAKE_NAMED_ARGUMENT(params);
IGOR_MAKE_NAMED_ARGUMENT(n_neighbors);
IGOR_MAKE_NAMED_ARGUMENT(return_distance);

/// @endcond

namespace detail {

/**
 * @brief Trait to check whether @p Args only contains named-parameter.
 */
template <typename... Args>
constexpr bool has_only_named_args_v = !igor::has_unnamed_arguments<Args...>();

}  // namespace detail

/**
 * @brief A small wrapper struct encapsulating all options that control the locality sensitive hashing behavior.
 */
struct locality_sensitive_hashing_options {
    /// The sued hash function type.
    hash_function_type hash_function = hash_function_type::random_projections;
    /// The number of hash functions in the hash pool.
    index_type hash_pool_size{};
    /// The number of hash functions per hash table.
    index_type num_hash_functions{};
    /// The number of used hash tables.
    index_type num_hash_tables{};
    /// The size of each hash table.
    hash_value_type hash_table_size{};
    /// The segment size for the random projections hash functions: \f$h_{a, b} = \frac{a \cdot x + b}{w}\f$.
    real_type w{};
    /// The number of cut-off points for the entropy-based hash functions.
    index_type num_cut_off_points{};
};

/**
 * @brief Class containing and managing all compile time and runtime hyperparameters to change the behavior of the LSH algorithm.
 */
class options {
  public:
    /**
     * @brief Construct a @ref sycl_lsh::options class from the provided command line arguments.
     * @param[in] argc the number of command line arguments
     * @param[in] argv the provided command line arguments
     * @param[in] comm the used MPI communicator
     */
    options(int argc, char **argv, const mpi::communicator &comm);

    /// The number of nearest-neighbors to search for.
    index_type k{};
    /// The path to the input data file.
    std::string data_file{};
    /// The type of the file parser.
    mpi::file_parser_type file_parser{};

    /// The file to which the calculated nearest-neighbors should be saved to.
    std::optional<std::string> knn_save_file{};
    /// The file to which the calculated nearest-neighbors **distances** should be saved to.
    std::optional<std::string> knn_dist_save_file{};
    /// The file containing the correct nearest-neighbors for calculating the resulting recall.
    std::optional<std::string> evaluate_knn_file{};
    /// The file containing the correct nearest-neighbors **distances** for calculating the error ratio.
    std::optional<std::string> evaluate_knn_dist_file{};

    /// Various options accessible on the respective device.
    locality_sensitive_hashing_options lsh_options{};
};

/**
 * @brief Print all options set in @p opt to the output stream @p out.
 * @param[in,out] out the output stream
 * @param[in] opt the @ref sycl_lsh::options
 * @return the output stream
 */
std::ostream &operator<<(std::ostream &out, const options &opt);

}  // namespace sycl_lsh

template <>
struct fmt::formatter<sycl_lsh::options> : fmt::ostream_formatter { };

#endif  // SYCL_LSH_OPTIONS_HPP
