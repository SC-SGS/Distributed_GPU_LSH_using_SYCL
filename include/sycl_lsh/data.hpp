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
#include "sycl_lsh/detail/get_linear_id.hpp"         // forward declaration
#include "sycl_lsh/memory_layout.hpp"                // sycl_lsh::memory_layout
#include "sycl_lsh/mpi/communicator.hpp"             // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/file_parser/file_parser.hpp"  // sycl_lsh::mpi::make_file_parser
#include "sycl_lsh/mpi/logger.hpp"                   // sycl_lsh::mpi::logger

#include "sycl_lsh/options.hpp"  // sycl_lsh::options

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter
#include "mpi.h"      // MPI_Sendrecv_replace
#include "sycl/sycl.hpp"

#include <memory>   // std::unique_ptr
#include <ostream>  // std::ostream
#include <utility>  // std::move
#include <vector>   // std::vector

namespace sycl_lsh {

// forward declare data class
template <memory_layout layout, typename Options>
class data;

namespace detail {

/**
 * @brief Specialization of the @ref sycl_lsh::get_linear_id class for the @ref sycl_lsh::data class to convert a multidimensional
 *        index to a one-dimensional one.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 * @tparam Options the @ref sycl_lsh::options type
 */
template <memory_layout layout, typename Options>
struct get_linear_id<data<layout, Options>> {
    /// The used @ref sycl_lsh::data type.
    using data_type = data<layout, Options>;
    /// The used @ref sycl_lsh::data_attributes type.
    using data_attributes_type = typename data_type::data_attributes_type;

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
    [[nodiscard]]
    index_type operator()(const index_type point, const index_type dim, const data_attributes_type &attr) const noexcept {  // TODO
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
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 * @param[in] logger the used @ref sycl_lsh::mpi::logger
 * @return the @ref sycl_lsh::data object representing the used data set (`[[nodiscard]]`)
 */
template <memory_layout layout, hash_function_type hash_function_t>
[[nodiscard]] auto make_data(const options<hash_function_t> &opt, const mpi::communicator &comm, const mpi::logger &logger) {
    auto file_parser = mpi::make_file_parser<real_type>(opt.data_file, opt.file_parser, mpi::file::mode::read, comm, logger);
    return data<layout, options<hash_function_t>>(*file_parser, comm, logger);
}

/**
 * @brief Class which represents the used data set.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 * @tparam Options the used @ref sycl_lsh::options type
 */
template <memory_layout layout, typename Options>
class data {
  public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                type aliases                                                //
    // ---------------------------------------------------------------------------------------------------------- //
    /// The type of the @ref sycl_lsh::options object.
    using options_type = Options;

    /// The type of the @ref sycl_lsh::data_attributes object representing the attributes of the used data set.
    using data_attributes_type = data_attributes<layout>;

    /// The type of the device buffer used by SYCL.
    using device_buffer_type = sycl::buffer<real_type, 1>;
    /// The type of the host buffer used to hide the MPI communications.
    using host_buffer_type = std::vector<real_type>;

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
    [[nodiscard]] static constexpr memory_layout get_memory_layout() noexcept { return layout; }
    /**
     * @brief Return the @ref sycl_lsh::data_attributes object representing the attributes of the used data set.
     * @return the @ref sycl_lsh::data_attributes (`[[nodiscard]]`)
     */
    [[nodiscard]] data_attributes_type get_attributes() const noexcept { return data_attributes_; }

    /**
     * @brief Returns the device buffer used in the SYCL kernels.
     * @return the device buffer (`[[nodiscard]]`)
     */
    [[nodiscard]] device_buffer_type &get_device_buffer() noexcept { return device_buffer_; }
    /**
     * @brief Returns the host buffer used to hide the MPI communication.
     * @return the host buffer (`[[nodiscard]]`)
     */
    [[nodiscard]] host_buffer_type &get_host_buffer() noexcept { return host_buffer_; }

  private:
    // befriend the factory function
    friend auto make_data<layout, options_type::used_hash_function_type>(const options_type &, const mpi::communicator &, const mpi::logger &);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::data object representing the used data set parsed by the file @p parser.
     * @param[in] parser the file parser used to parse the given data file
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     */
    data(const mpi::file_parser<real_type> &parser, const mpi::communicator &comm, const mpi::logger &logger);

    /// The associated MPI communicator.
    const mpi::communicator &comm_;
    /// The associated data attributes.
    const data_attributes_type data_attributes_;

    /// The SYCL device buffer.
    device_buffer_type device_buffer_;
    /// The SYCL host buffer
    host_buffer_type host_buffer_;
};

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout, typename Options>
data<layout, Options>::data(const mpi::file_parser<real_type> &parser,
                            const mpi::communicator &comm,
                            const mpi::logger &logger) : comm_{ comm },
                                                         data_attributes_{ parser.parse_total_size(), parser.parse_rank_size(), parser.parse_dims() },
                                                         device_buffer_(data_attributes_.rank_size * data_attributes_.dims),
                                                         host_buffer_(parser.parse_content()) {
    const mpi::timer mpi_timer{ comm_ };

    // change memory layout from aos to soa if requested
    if constexpr (layout == memory_layout::soa) {
        host_buffer_type soa_host_buffer(data_attributes_.rank_size * data_attributes_.dims);
        data_attributes<memory_layout::aos> parsed_data_attributes{ data_attributes_ };

        const detail::get_linear_id<data<memory_layout::soa, options_type>> get_linear_id_soa{};

        for (index_type point = 0; point < data_attributes_.rank_size; ++point) {
            for (index_type dim = 0; dim < data_attributes_.dims; ++dim) {
                soa_host_buffer[get_linear_id_soa(point, dim, data_attributes_)] = host_buffer_[point * data_attributes_.dims + dim];
            }
        }

        host_buffer_ = std::move(soa_host_buffer);
    }

    // copy data to device buffer
    auto acc = device_buffer_.template get_access<sycl::access::mode::discard_write>();
    for (index_type i = 0; i < acc.size(); ++i) {
        acc[i] = host_buffer_[i];
    }

    logger.log("Created data object in {}.\n", mpi_timer.elapsed());
}

// ---------------------------------------------------------------------------------------------------------- //
//                                             update host buffer                                             //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout, typename Options>
void data<layout, Options>::send_receive_host_buffer() {
    const int destination = (comm_.rank() + 1) % comm_.size();
    const int source = (comm_.size() + (comm_.rank() - 1) % comm_.size()) % comm_.size();

    SYCL_LSH_MPI_ERROR_CHECK(MPI_Sendrecv_replace(host_buffer_.data(), host_buffer_.size(), mpi::detail::mpi_datatype<typename host_buffer_type::value_type>(), destination, 0, source, 0, comm_.get(), MPI_STATUS_IGNORE));
}

// ---------------------------------------------------------------------------------------------------------- //
//                                            output stream overload                                          //
// ---------------------------------------------------------------------------------------------------------- //
/**
 * @brief Prints all attributes set in the @ref sycl_lsh::data_attributes associated with @p data to the output stream @p out.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 * @tparam Options the used @ref sycl_lsh::options type
 * @param[in,out] out the output stream
 * @param data the @ref sycl_lsh::data object representing the used data set
 * @return the output stream
 */
template <memory_layout layout, typename Options>
std::ostream &operator<<(std::ostream &out, const data<layout, Options> &data) {
    return out << data.get_attributes();
}

}  // namespace sycl_lsh

template <sycl_lsh::memory_layout layout, typename Options>
struct fmt::formatter<sycl_lsh::data<layout, Options>> : fmt::ostream_formatter {};

#endif  // SYCL_LSH_DATA_HPP
