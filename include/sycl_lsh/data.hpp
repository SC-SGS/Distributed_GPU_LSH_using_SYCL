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

#include "sycl_lsh/constants.hpp"                    // sycl_lsh::{real_type, index_type, hash_value_type}
#include "sycl_lsh/data_attributes.hpp"              // sycl_lsh::data_attributes
#include "sycl_lsh/detail/device_ptr.hpp"            // sycl_lsh::detail::device_ptr
#include "sycl_lsh/detail/get_linear_id.hpp"         // forward declaration
#include "sycl_lsh/memory_layout.hpp"                // sycl_lsh::memory_layout
#include "sycl_lsh/mpi/communicator.hpp"             // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/file_parser/file_parser.hpp"  // sycl_lsh::mpi::make_file_parser
#include "sycl_lsh/mpi/logger.hpp"                   // sycl_lsh::mpi::logger
#include "sycl_lsh/options.hpp"                      // sycl_lsh::options

#include "sycl/sycl.hpp"

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter
#include "mpi.h"          // MPI_Sendrecv_replace

#include <memory>   // std::unique_ptr
#include <ostream>  // std::ostream
#include <utility>  // std::move
#include <vector>   // std::vector

namespace sycl_lsh {

// forward declare data class
template <memory_layout layout>
class data;

namespace detail {

/**
 * @brief Specialization of the @ref sycl_lsh::get_linear_id class for the @ref sycl_lsh::data class to convert a multidimensional
 *        index to a one-dimensional one.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 */
template <memory_layout layout>
struct get_linear_id<data<layout>> {
    /// The used @ref sycl_lsh::data type.
    using data_type = data<layout>;

    /**
     * @brief Convert the multidimensional index to a one-dimensional index.
     * @param[in] point the requested data point
     * @param[in] dim the requested dimension of @p point
     * @param[in] attr the attributes of the used data set
     * @return the one-dimensional index (`[[nodiscard]]`)
     *
     * @pre @p point must be in the range `[0, number of data points on the current MPI rank)` (currently disabled).
     * @pre @p dim must be in the range `[0, number of dimensions per data point)` (currently disabled).
     */
    [[nodiscard]] index_type operator()(const index_type point, const index_type dim, const data_attributes &attr) const noexcept {  // TODO
        // SYCL_LSH_ASSERT(0 <= point && point < attr.rank_size, "Out-of-bounce access for data point!\n");
        // SYCL_LSH_ASSERT(0 <= dim && dim < attr.dims, "Out-of-bounce access for dimension!\n");

        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return point * attr.dims + dim;
        } else {
            // Struct of Arrays
            return point + dim * attr.rank_size;
        }
    }
};

}  // namespace detail

/**
 * @brief Factory function for the @ref sycl_lsh::data class.
 * @details Used to be able to automatically deduce the @ref sycl_lsh::options type.
 * @tparam layout the used @ref sycl_lsh::memory_layout type
 * @param[in] opt the used @ref sycl_lsh::options
 * @param[in] queue the SYCL queue to run on
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 * @param[in] logger the used @ref sycl_lsh::mpi::logger
 * @return the @ref sycl_lsh::data object representing the used data set (`[[nodiscard]]`)
 */
template <memory_layout layout>
[[nodiscard]] auto make_data(const options &opt, sycl::queue &queue, const mpi::communicator &comm, const mpi::logger &logger) {
    auto file_parser = mpi::make_file_parser<real_type>(opt.data_file, opt.file_parser, mpi::file::mode::read, comm, logger);
    return data<layout>(*file_parser, queue, comm, logger);
}

/**
 * @brief Class which represents the used data set.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 */
template <memory_layout layout>
class data {
  public:
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
     * @brief Returns the specified @ref sycl_lsh::memory_layout type.
     * @return the @ref sycl_lsh::memory_layout type (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr static memory_layout get_memory_layout() noexcept { return layout; }

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
    [[nodiscard]] std::vector<real_type> &get_host_buffer() noexcept { return host_buffer_; }

  private:
    // befriend the factory function
    friend auto make_data<layout>(const options &, sycl::queue &, const mpi::communicator &, const mpi::logger &);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::data object representing the used data set parsed by the file @p parser.
     * @param[in] parser the file parser used to parse the given data file
     * @param[in] queue the SYCL queue to run on
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     */
    data(const mpi::file_parser<real_type> &parser, sycl::queue &queue, const mpi::communicator &comm, const mpi::logger &logger);

    /// The associated SYCL queue representing the device to run on.
    sycl::queue &queue_;

    /// The associated MPI communicator.
    const mpi::communicator &comm_;
    /// The associated data attributes.
    const data_attributes data_attributes_;

    /// The SYCL device buffer.
    detail::device_ptr<real_type> device_ptr_{};
    /// The host buffer,
    std::vector<real_type> host_buffer_;
};

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout>
data<layout>::data(const mpi::file_parser<real_type> &parser,
                   sycl::queue &queue,
                   const mpi::communicator &comm,
                   const mpi::logger &logger) :
    queue_{ queue },
    comm_{ comm },
    data_attributes_{ parser.parse_total_size(), parser.parse_rank_size(), parser.parse_dims() },
    host_buffer_(parser.parse_content()) {
    const mpi::timer mpi_timer{ comm_ };

    // change memory layout from aos to soa if requested
    if constexpr (layout == memory_layout::soa) {
        std::vector<real_type> soa_host_buffer(data_attributes_.rank_size * data_attributes_.dims);

        const detail::get_linear_id<data<memory_layout::soa>> get_linear_id_soa{};

        for (index_type point = 0; point < data_attributes_.rank_size; ++point) {
            for (index_type dim = 0; dim < data_attributes_.dims; ++dim) {
                soa_host_buffer[get_linear_id_soa(point, dim, data_attributes_)] = host_buffer_[point * data_attributes_.dims + dim];
            }
        }

        host_buffer_ = std::move(soa_host_buffer);
    }

    // allocate memory on the device and copy the data over
    device_ptr_ = detail::device_ptr<real_type>{ detail::shape{ data_attributes_.rank_size, data_attributes_.dims }, queue_ };
    device_ptr_.copy_to_device(host_buffer_);

    logger.log("Created data object in {}.\n", mpi_timer.elapsed());
}

// ---------------------------------------------------------------------------------------------------------- //
//                                             update host buffer                                             //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout>
void data<layout>::send_receive_host_buffer() {
    const int destination = (comm_.rank() + 1) % comm_.size();
    const int source = (comm_.size() + (comm_.rank() - 1) % comm_.size()) % comm_.size();

    SYCL_LSH_MPI_ERROR_CHECK(MPI_Sendrecv_replace(host_buffer_.data(), host_buffer_.size(), mpi::detail::mpi_datatype<real_type>(), destination, 0, source, 0, comm_.get(), MPI_STATUS_IGNORE));
}

// ---------------------------------------------------------------------------------------------------------- //
//                                            output stream overload                                          //
// ---------------------------------------------------------------------------------------------------------- //
/**
 * @brief Prints all attributes set in the @ref sycl_lsh::data_attributes associated with @p data to the output stream @p out.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 * @param[in,out] out the output stream
 * @param data the @ref sycl_lsh::data object representing the used data set
 * @return the output stream
 */
template <memory_layout layout>
std::ostream &operator<<(std::ostream &out, const data<layout> &data) {
    return out << data.get_attributes();
}

}  // namespace sycl_lsh

template <sycl_lsh::memory_layout layout>
struct fmt::formatter<sycl_lsh::data<layout>> : fmt::ostream_formatter { };

#endif  // SYCL_LSH_DATA_HPP
