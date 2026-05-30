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

#include "sycl_lsh/constants.hpp"                    // sycl_lsh::real_type
#include "sycl_lsh/data_attributes.hpp"              // sycl_lsh::data_attributes
#include "sycl_lsh/detail/device_ptr.hpp"            // sycl_lsh::detail::device_ptr
#include "sycl_lsh/mpi/communicator.hpp"             // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/file_parser/file_parser.hpp"  // sycl_lsh::mpi::make_file_parser
#include "sycl_lsh/mpi/logger.hpp"                   // sycl_lsh::mpi::logger
#include "sycl_lsh/options.hpp"                      // sycl_lsh::options

#include "sycl/sycl.hpp"

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include "matrix.hpp"
#include <iosfwd>  // std::ostream forward declaration

namespace sycl_lsh {

/**
 * @brief Class which represents the used data set.
 */
class data_set {
  public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::data object representing the used data set parsed by the file @p parser.
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] queue the SYCL queue to run on
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     */
    data_set(const options &opt, sycl::queue &queue, const mpi::communicator &comm, const mpi::logger &logger);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                             update host buffer                                             //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Send the elements of the host buffer to the neighboring host buffer replacing its content using a ring like send pattern.
     */
    void send_receive_host_buffer();

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                   getter                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Return the @ref sycl_lsh::data_attributes object representing the attributes of the used data set.
     * @return the @ref sycl_lsh::data_attributes (`[[nodiscard]]`)
     */
    [[nodiscard]] data_attributes get_attributes() const noexcept { return data_attributes_; }

    /**
     * @brief Returns the device_ptr wrapping the device memory used in the SYCL kernels.
     * @return the device memory (`[[nodiscard]]`)
     */
    [[nodiscard]] const detail::device_ptr<real_type> &get_device_ptr() const noexcept { return device_ptr_; }

    /**
     * @brief Returns the device_ptr wrapping the device memory used in the SYCL kernels.
     * @return the device memory (`[[nodiscard]]`)
     */
    [[nodiscard]] detail::device_ptr<real_type> &get_device_ptr() noexcept { return device_ptr_; }

    /**
     * @brief Returns the host buffer used to hide the MPI communication.
     * @return the host buffer (`[[nodiscard]]`)
     */
    [[nodiscard]] aos_matrix<real_type> &get_host_buffer() noexcept { return data_; }

  private:
    /// The associated SYCL queue representing the device to run on.
    sycl::queue &queue_;
    /// The associated MPI communicator.
    mpi::communicator comm_;

    /// The associated data attributes.
    data_attributes data_attributes_{};
    /// The host buffer represented as a matrix.
    aos_matrix<real_type> data_{};

    /// The SYCL device buffer.
    detail::device_ptr<real_type> device_ptr_{};
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
