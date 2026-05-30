/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the @ref sycl_lsh::data class representing the used data set.
 */

#ifndef SYCL_LSH_DATA_HPP
#define SYCL_LSH_DATA_HPP
#pragma once

#include "sycl_lsh/constants.hpp"         // sycl_lsh::real_type
#include "sycl_lsh/data_attributes.hpp"   // sycl_lsh::data_attributes
#include "sycl_lsh/matrix.hpp"            // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/logger.hpp"        // sycl_lsh::mpi::logger
#include "sycl_lsh/options.hpp"           // sycl_lsh::options

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // std::ostream forward declaration

namespace sycl_lsh {

/**
 * @brief Class which represents the used data set.
 */
class data_set {
    // befriend hash_tables class
    template <typename>
    friend class hash_tables;

  public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::data object representing the used data set parsed by the file @p parser.
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     */
    data_set(const options &opt, const mpi::communicator &comm, const mpi::logger &logger);

    /**
     * @brief Return the data points in this data set.
     * @return the data points (`[[nodiscard]]`)
     */
    [[nodiscard]] const aos_matrix<real_type> &data() const { return *data_ptr_; }

    /**
     * @brief Return the data attributes of this data set.
     * @return the data set attributes (`[[nodiscard]]`)
     */
    [[nodiscard]] data_attributes attributes() const noexcept { return data_attributes_; }

  private:
    // Modifying getter. Only used in the hash_tables class for the send_receive_round_robin implementation.
    [[nodiscard]] aos_matrix<real_type> &mutable_data() { return *data_ptr_; }

    /// The associated data attributes.
    data_attributes data_attributes_{};

    /// The host buffer represented as a matrix.
    std::shared_ptr<aos_matrix<real_type>> data_ptr_{ nullptr };
};

/**
 * @brief Prints all attributes set in the @ref sycl_lsh::data_attributes associated with @p data to the output stream @p out.
 * @param[in,out] out the output stream
 * @param data the @ref sycl_lsh::data object representing the used data set
 * @return the output stream
 */
std::ostream &operator<<(std::ostream &out, const data_set &data);

}  // namespace sycl_lsh

template <>
struct fmt::formatter<sycl_lsh::data_set> : fmt::ostream_formatter { };

#endif  // SYCL_LSH_DATA_HPP
