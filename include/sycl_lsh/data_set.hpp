/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the @ref sycl_lsh::data_set class representing the used data set.
 */

#ifndef SYCL_LSH_DATA_SET_HPP
#define SYCL_LSH_DATA_SET_HPP
#pragma once

#include "sycl_lsh/constants.hpp"         // sycl_lsh::real_type
#include "sycl_lsh/detail/utility.hpp"    // SYCL_LSH_REQUIRES
#include "sycl_lsh/matrix.hpp"            // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator
#include "sycl_lsh/options.hpp"           // sycl_lsh::options, sycl_lsh::detail::has_only_named_args_v

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter
#include "igor/igor.hpp"  // igor::parser

#include <iosfwd>  // std::ostream forward declaration
#include <string>  // std::string

namespace sycl_lsh {

// forward declare profiler class (to break circular dependency)
class profiler;

namespace detail::hashing {

// forward declare hash_tables class
template <typename>
class hash_tables;

}  // namespace detail::hashing

/**
 * @brief Class which represents the used data set.
 */
class data_set {
    // befriend hash_tables class
    template <typename>
    friend class detail::hashing::hash_tables;

  public:
    /**
     * @brief Small helper struct encapsulating all data set attributes.
     */
    struct attributes {
        /// The **total** number of data points of the used @ref sycl_lsh::data_set.
        index_type total_size{ 0 };
        /// The number of data points on **the current** MPI rank.
        index_type rank_size{ 0 };
        /// The number of dimensions of each data point of the used @ref sycl_lsh::data_set.
        index_type dims{ 0 };
    };

    /**
     * @brief Default construct an empty @ref sycl_lsh::data_set.
     */
    data_set() = default;

    /**
     * @brief Construct a new @ref sycl_lsh::data_set from @p filename using the @p file_parser type.
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] filename the file to parse
     * @param[in] file_parser the @ref sycl_lsh::mpi::file_parser_type
     * @param[in] named_args optional additional named arguments (sycl_lsh::profiler)
     */
    template <typename... NamedArgs, SYCL_LSH_REQUIRES(detail::has_only_named_args_v<NamedArgs...>)>
    data_set(const mpi::communicator &comm, const std::string &filename, const mpi::file_parser_type file_parser, NamedArgs &&...named_args) {
        // check igor parameter
        const igor::parser parser{ std::forward<NamedArgs>(named_args)... };

        // check whether a performance profiler has been provided
        if constexpr (parser.has(sycl_lsh::perf_profiler)) {
            // update the profiler
            profiler_ = static_cast<decltype(profiler_)>(parser(sycl_lsh::perf_profiler));
        }

        // initialize data
        this->init(comm, filename, file_parser);
    }

    /**
     * @brief Construct a new @ref sycl_lsh::data_set from @p filename using the @p file_parser type.
     * @details Uses the @ref sycl_lsh::mpi::detail::binary_parser by default.
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] filename the file to parse
     * @param[in] named_args optional additional named arguments (sycl_lsh::profiler)
     */
    template <typename... NamedArgs, SYCL_LSH_REQUIRES(detail::has_only_named_args_v<NamedArgs...>)>
    data_set(const mpi::communicator &comm, const std::string &filename, NamedArgs &&...named_args) :
        data_set{ comm, filename, mpi::file_parser_type::binary, std::forward<NamedArgs>(named_args)... } { }

    /**
     * @brief Return the data points in this @ref sycl_lsh::data_set.
     * @return the data points (`[[nodiscard]]`)
     */
    [[nodiscard]] const aos_matrix<real_type> &data() const { return *data_ptr_; }

    /**
     * @brief Return the data attributes of this @ref sycl_lsh::data_set.
     * @return the @ref sycl_lsh::data_set::attributes (`[[nodiscard]]`)
     */
    [[nodiscard]] attributes get_attributes() const noexcept { return attributes_; }

  private:
    /**
     * @brief Initialize the data from @p filename using the @p file_parser type.
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] filename the file to parse
     * @param[in] file_parser the @ref sycl_lsh::mpi::file_parser_type
     */
    void init(const mpi::communicator &comm, const std::string &filename, mpi::file_parser_type file_parser);

    /**
     * @brief Return the data points in this @ref sycl_lsh::data_set in a mutable way.
     * @attention Must be used with caution!
     * @return the data points (`[[nodiscard]]`)
     */
    [[nodiscard]] aos_matrix<real_type> &mutable_data() { return *data_ptr_; }

    /// The associated @ref sycl_lsh::data_set::attributes.
    attributes attributes_{};

    /// The host buffer represented as a @ref sycl_lsh::aos_matrix.
    std::shared_ptr<aos_matrix<real_type>> data_ptr_{ nullptr };

    /// The optional @ref sycl_lsh::profiler.
    std::shared_ptr<profiler> profiler_{ nullptr };
};

/**
 * @brief Prints all attributes set in the @ref sycl_lsh::data_set::attributes associated with @p data to the output stream @p out.
 * @param[in,out] out the output stream
 * @param data the @ref sycl_lsh::data_set object representing the used data set
 * @return the output stream
 */
std::ostream &operator<<(std::ostream &out, const data_set &data);

}  // namespace sycl_lsh

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<sycl_lsh::data_set> : fmt::ostream_formatter { };

/// @endcond

#endif  // SYCL_LSH_DATA_SET_HPP
