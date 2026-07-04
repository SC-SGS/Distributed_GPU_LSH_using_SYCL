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

#include "sycl_lsh/constants.hpp"              // sycl_lsh::real_type, sycl_lsh::index_type, sycl_lsh::hash_value_type
#include "sycl_lsh/hash_function_types.hpp"    // sycl_lsh::hash_function_type
#include "sycl_lsh/mpi/communicator.hpp"       // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/logging.hpp"     // sycl_lsh::mpi::detail::log
#include "sycl_lsh/mpi/file_parser_types.hpp"  // sycl_lsh::mpi::file_parser
#include "sycl_lsh/profiling_types.hpp"        // sycl_lsh::profiling_types

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter
#include "igor/igor.hpp"  // IGOR_MAKE_NAMED_ARGUMENT, igor::parser, igor::has_unnamed_arguments, igor::has_other_than

#include <cstddef>   // std::size_t
#include <iosfwd>    // std::ostream  forward declarations
#include <optional>  // std::optional
#include <string>    // std::string

namespace sycl_lsh {

/// @cond Doxygen_suppress

// create named arguments
IGOR_MAKE_NAMED_ARGUMENT(lsh_options);
IGOR_MAKE_NAMED_ARGUMENT(n_neighbors);
IGOR_MAKE_NAMED_ARGUMENT(return_distance);
IGOR_MAKE_NAMED_ARGUMENT(perf_profiler);
IGOR_MAKE_NAMED_ARGUMENT(work_group_size);

/// @endcond

namespace detail {

/**
 * @brief Trait to check whether @p Args only contains named-parameter.
 */
template <typename... Args>
constexpr bool has_only_named_args_v = !igor::has_unnamed_arguments<Args...>();

}  // namespace detail

/**
 * @brief A small wrapper struct encapsulating all options that control the Locality Sensitive Hashing behavior.
 */
struct locality_sensitive_hashing_options {
    /// The used @ref sycl_lsh::hash_function_type.
    hash_function_type hash_function = hash_function_type::random_projections;
    /// The number of hash functions in the hash pool.
    index_type hash_pool_size{};
    /// The number of hash functions per hash table.
    index_type num_hash_functions{};
    /// The number of used hash tables.
    index_type num_hash_tables{};
    /// The size of each hash table.
    hash_value_type hash_table_size{};
    /// The segment size for the @ref sycl_lsh::detail::hashing::random_projections hash functions: \f$h_{a, b} = \frac{a \cdot x + b}{w}\f$.
    real_type w{};
    /// The number of cut-off points for the @ref sycl_lsh::detail::hashing::entropy_based hash functions.
    index_type num_cut_off_points{};
};

namespace detail {

/**
 * @details Perform some sanity checks for the Locality Sensitive Hashing options.
 * @param options the LSH options to sanity check
 *
 * @throws sycl_lsh::invalid_lsh_option_exception if @p options.hash_pool_size is smaller than 1
 * @throws sycl_lsh::invalid_lsh_option_exception if @p options.num_hash_functions is smaller than 1
 * @throws sycl_lsh::invalid_lsh_option_exception if @p options.num_hash_tables is smaller than 1
 * @throws sycl_lsh::invalid_lsh_option_exception if @p options.hash_table_size is smaller than 1
 * @throws sycl_lsh::invalid_lsh_option_exception if @p options.w is smaller than 1
 * @throws sycl_lsh::invalid_lsh_option_exception if @p options.num_cut_off_points is smaller than 1
 */
void sanity_check_locality_sensitive_hashing_options(const locality_sensitive_hashing_options &options);

}  // namespace detail

/**
 * @brief Class containing and managing all compile time and runtime hyperparameters to change the behavior of the LSH algorithm.
 */
class options {
  public:
    /**
     * @brief Construct a @ref sycl_lsh::options class from the provided command line arguments.
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] argc the number of command line arguments
     * @param[in] argv the provided command line arguments
     */
    options(const mpi::communicator &comm, int &argc, char **&argv);

    /// The number of nearest-neighbors to search for.
    index_type n_neighbors{};
    /// The path to the input data file.
    std::string data_file{};
    /// The type of the file parser.
    mpi::file_parser_type file_parser{};

    /// The type of the used profiling.
    profiling_types profiling_type{};
    /// The output file to save the profiling information to. Has no effect if @ref sycl_lsh::options::profiling_type is set to @ref sycl_lsh::profiling_types::none .
    std::optional<std::string> profiling_file{};

    /// The file to which the calculated nearest-neighbors should be saved to.
    std::optional<std::string> indices_save_file{};
    /// The file to which the calculated nearest-neighbors **distances** should be saved to.
    std::optional<std::string> distances_save_file{};
    /// The file containing the correct nearest-neighbors for calculating the resulting recall.
    std::optional<std::string> indices_ground_truth_file{};
    /// The file containing the correct nearest-neighbors **distances** for calculating the error ratio.
    std::optional<std::string> distances_ground_truth_file{};

    /// The number of work-items per work-group for the main kernels. Used to fine-tune the performance.
    std::size_t work_group_size{};

    /// Various options accessible on the respective device.
    locality_sensitive_hashing_options lsh_options{};
};

/**
 * @brief Print all LSH related options set in @p opt to the output stream @p out and add the @p prefix to each newline.
 * @param[in,out] out the output stream
 * @param[in] opt the @ref sycl_lsh::locality_sensitive_hashing_options
 * @param[in] prefix the prefix to add to each newline
 * @return the output stream
 */
std::ostream &output_with_prefix(std::ostream &out, const locality_sensitive_hashing_options &opt, const std::string_view &prefix = "");

/**
 * @brief Print all LSH related options set in @p opt to the output stream @p out.
 * @param[in,out] out the output stream
 * @param[in] opt the @ref sycl_lsh::locality_sensitive_hashing_options
 * @return the output stream
 */
std::ostream &operator<<(std::ostream &out, const locality_sensitive_hashing_options &opt);

/**
 * @brief Print all options set in @p opt to the output stream @p out.
 * @param[in,out] out the output stream
 * @param[in] opt the @ref sycl_lsh::options
 * @return the output stream
 */
std::ostream &operator<<(std::ostream &out, const options &opt);

}  // namespace sycl_lsh

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<sycl_lsh::locality_sensitive_hashing_options> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<sycl_lsh::options> : fmt::ostream_formatter { };

/// @endcond

#endif  // SYCL_LSH_OPTIONS_HPP
