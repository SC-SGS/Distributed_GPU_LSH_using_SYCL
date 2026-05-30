/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the @ref sycl_lsh::data class representing the used data set.
 */

#ifndef SYCL_LSH_DATA_SET_HPP
#define SYCL_LSH_DATA_SET_HPP
#pragma once

#include "sycl_lsh/constants.hpp"         // sycl_lsh::real_type
#include "sycl_lsh/matrix.hpp"            // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator
#include "sycl_lsh/options.hpp"           // sycl_lsh::options

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // std::ostream forward declaration

namespace sycl_lsh {

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
        /// The **total** number of data points of the used data set.
        index_type total_size{ 0 };
        /// The number of data points on **the current** MPI rank.
        index_type rank_size{ 0 };
        /// The number of dimensions of each data point of the used data set.
        index_type dims{ 0 };
    };

    /**
     * @brief Default construct an empty data set.
     */
    data_set() = default;
    /**
     * @brief Construct a new @ref sycl_lsh::data object representing the used data set parsed by the file @p parser.
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     */
    data_set(const options &opt, const mpi::communicator &comm);

    /**
     * @brief Return the data points in this data set.
     * @return the data points (`[[nodiscard]]`)
     */
    [[nodiscard]] const aos_matrix<real_type> &data() const { return *data_ptr_; }

    /**
     * @brief Return the data attributes of this data set.
     * @return the data set attributes (`[[nodiscard]]`)
     */
    [[nodiscard]] attributes get_attributes() const noexcept { return attributes_; }

  private:
    // Modifying getter. Only used in the hash_tables class for the send_receive_round_robin implementation.
    [[nodiscard]] aos_matrix<real_type> &mutable_data() { return *data_ptr_; }

    /// The associated data attributes.
    attributes attributes_{};

    /// The host buffer represented as a matrix.
    std::shared_ptr<aos_matrix<real_type>> data_ptr_{ nullptr };
};

/**
 * @brief Prints all attributes set in the @ref sycl_lsh::data_set::attributes associated with @p data to the output stream @p out.
 * @param[in,out] out the output stream
 * @param data the @ref sycl_lsh::data object representing the used data set
 * @return the output stream
 */
std::ostream &operator<<(std::ostream &out, const data_set &data);

}  // namespace sycl_lsh

template <>
struct fmt::formatter<sycl_lsh::data_set> : fmt::ostream_formatter { };

#endif  // SYCL_LSH_DATA_SET_HPP
