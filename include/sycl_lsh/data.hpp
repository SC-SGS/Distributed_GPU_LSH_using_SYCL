/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-08
 *
 * @brief Implements the @ref sycl_lsh::data class representing the used data set.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP

#include <sycl_lsh/argv_parser.hpp>
#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/get_linear_id.hpp>
#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/memory_layout.hpp>
#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/file.hpp>
#include <sycl_lsh/mpi/file_parser/file_parser.hpp>
#include <sycl_lsh/mpi/logger.hpp>
#include <sycl_lsh/options.hpp>
#include <sycl_lsh/data_attributes.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <mpi.h>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace sycl_lsh {

    // forward declare data class
    template <memory_layout layout, typename Options>
    class data;

    /**
     * @brief Factory function for the @ref sycl_lsh::data class.
     * @details Used to be able to automatically deduce the @ref sycl_lsh::options type.
     * @tparam layout the used @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     * @param[in] parser the used @ref sycl_lsh::argv_parser
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     * @return the @ref sycl_lsh::data object representing the used data set (`[[nodiscard]]`)
     */
    template <memory_layout layout, typename Options>
    [[nodiscard]]
    inline auto make_data(const argv_parser& parser, const Options&, const mpi::communicator& comm, const mpi::logger& logger) {
        using options_type = Options;
        using real_type = typename options_type::real_type;
        auto file_parser = mpi::make_file_parser<real_type, options_type>(parser.argv_as<std::string>("data_file"), parser, mpi::file::mode::read, comm, logger);
        return data<layout, options_type>(*file_parser, comm, logger);
    }

    /**
     * @brief Specialization of the @ref sycl_lsh::get_linear_id class for the @ref sycl_lsh::data class to convert a multi-dimensional
     *        index to an one-dimensional one.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the @ref sycl_lsh::options type
     */
    template <memory_layout layout, typename Options>
    struct get_linear_id<data<layout, Options>> {

        /// The used @ref sycl_lsh::data type.
        using data_type = data<layout, Options>;
        /// The used integral type (used for indices).
        using index_type = typename data_type::index_type;
        /// The used @ref sycl_lsh::data_attributes type.
        using data_attributes_type = typename data_type::data_attributes_type;

        /**
         * @brief Convert the multi-dimensional index to an one-dimensional index.
         * @param[in] point the requested data point
         * @param[in] dim the requested dimension of @p point
         * @param[in] attr the attributes of the used data set
         * @return the one-dimensional index (`[[nodiscard]]`)
         *
         * @pre @p point must be in the range `[0, number of data points on the current MPI rank)` (currently disabled).
         * @pre @p dim must be in the range `[0, number of dimensions per data point)` (currently disabled).
         */
        [[nodiscard]]
        index_type operator()(const index_type point, const index_type dim, const data_attributes_type& attr) const noexcept {
//            SYCL_LSH_DEBUG_ASSERT(0 <= point && point < attr.rank_size, "Out-of-bounce access for data point!\n");
//            SYCL_LSH_DEBUG_ASSERT(0 <= dim && dim < attr.dims, "Out-of-bounce access for dimension!\n");

            if constexpr (layout == memory_layout::aos) {
                // Array of Structs
                return point * attr.dims + dim;
            } else {
                // Struct of Arrays
                return point + dim * attr.rank_size;
            }
        }

    };


    /**
     * @brief Class which represents the used data set.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     */
    template <memory_layout layout, typename Options>
    class data : private detail::data_base {
        // ---------------------------------------------------------------------------------------------------------- //
        //                                      template parameter sanity checks                                      //
        // ---------------------------------------------------------------------------------------------------------- //
        static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must be a sycl_lsh::options type!");
    public:
        // ---------------------------------------------------------------------------------------------------------- //
        //                                                type aliases                                                //
        // ---------------------------------------------------------------------------------------------------------- //
        /// The type of the @ref sycl_lsh::options object.
        using options_type = Options;
        /// The used floating point type for the data points.
        using real_type = typename options_type::real_type;
        /// The used integral type for indices.
        using index_type = typename options_type::index_type;
        
        /// The type of the @ref sycl_lsh::data_attributes object representing the attributes of the used data set.
        using data_attributes_type = data_attributes<layout, index_type>;

        /// The type of the device buffer used by SYCL.
        using device_buffer_type = sycl::buffer<real_type, 1>;
        /// The type of the host buffer used to hide the MPI communications.
        using host_buffer_type = std::vector<real_type>;


        // ---------------------------------------------------------------------------------------------------------- //
        //                                             update host buffer                                             //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Send the elements of the active buffer to the neighboring inactive buffer using a ring like send pattern.
         * @details Swaps the active and inactive host buffers.
         */
        void send_receive_host_buffer();


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                   getter                                                   //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Returns the specified @ref sycl_lsh::memory_layout type.
         * @return the @ref sycl_lsh::memory_layout type (`[[nodiscard]]`)
         */
        [[nodiscard]]
        constexpr memory_layout get_memory_layout() const noexcept { return layout; }
        /**
         * @brief Return the @ref sycl_lsh::data_attributes object representing the attributes of the used data set.
         * @return the @ref sycl_lsh::data_attributes (`[[nodiscard]]`)
         */
        [[nodiscard]]
        data_attributes_type get_attributes() const noexcept { return data_attributes_; }

        /**
         * @brief Returns the device buffer used in the SYCL kernels.
         * @return the device buffer (`[[nodiscard]]`)
         */
        [[nodiscard]]
        device_buffer_type& get_device_buffer() noexcept { return device_buffer_; }
        /**
         * @brief Returns the host buffer used to hide the MPI communication.
         * @return the host buffer (`[[nodiscard]]`)
         */
        [[nodiscard]]
        host_buffer_type& get_host_buffer() noexcept { return host_buffer_active_; }

    private:
        // befriend the factory function
        friend auto make_data<layout, Options>(const argv_parser&, const Options&, const mpi::communicator&, const mpi::logger&);

        // ---------------------------------------------------------------------------------------------------------- //
        //                                                constructor                                                 //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Construct a new @ref sycl_lsh::data object representing the used data set parsed by the file parser @p parser.
         * @param[in] parser the file parser used to parse the given data file
         * @param[in] opt the used @ref sycl_lsh::options
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         * @param[in] logger the used @ref sycl_lsh::mpi::logger
         */
        data(const mpi::file_parser<options_type, real_type>& parser, const mpi::communicator& comm, const mpi::logger& logger);

        const mpi::communicator& comm_;

        const data_attributes_type data_attributes_;

        device_buffer_type device_buffer_;
        host_buffer_type host_buffer_active_;
        host_buffer_type host_buffer_inactive_;
    };
    

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
    std::ostream& operator<<(std::ostream& out, const data<layout, Options>& data) {
        return out << data.get_attributes();
    }


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options>
    data<layout, Options>::data(const mpi::file_parser<Options, typename Options::real_type>& parser,
                                const mpi::communicator& comm,
                                const mpi::logger& logger)
            : comm_(comm),
              data_attributes_(parser.parse_total_size(), parser.parse_rank_size(), parser.parse_dims()),
              device_buffer_(data_attributes_.rank_size * data_attributes_.dims),
              host_buffer_active_(parser.parse_content()),
              host_buffer_inactive_(data_attributes_.rank_size * data_attributes_.dims)
    {
        mpi::timer t(comm_);

        // change memory layout from aos to soa if requested
        if constexpr (layout == memory_layout::soa) {
            data_attributes<memory_layout::aos, index_type> parsed_data_attributes(data_attributes_);

            const get_linear_id<data<memory_layout::soa, options_type>> get_linear_id_soa;

            for (index_type point = 0; point < data_attributes_.rank_size; ++point) {
                for (index_type dim = 0; dim < data_attributes_.dims; ++dim) {
                    host_buffer_inactive_[get_linear_id_soa(point, dim, data_attributes_)]
                            = host_buffer_active_[point * data_attributes_.dims + dim];
                }
            }

            using std::swap;
            swap(host_buffer_active_, host_buffer_inactive_);
        }

        // copy data to device buffer
        auto acc = device_buffer_.template get_access<sycl::access::mode::discard_write>();
        for (index_type i = 0; i < acc.get_count(); ++i) {
            acc[i] = host_buffer_active_[i];
        }

        logger.log("Created data object in {}.\n", t.elapsed());
    }


    // ---------------------------------------------------------------------------------------------------------- //
    //                                             update host buffer                                             //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options>
    void data<layout, Options>::send_receive_host_buffer() {
        const int destination = (comm_.rank() + 1) % comm_.size();
        const int source = (comm_.size() + (comm_.rank() - 1) % comm_.size()) % comm_.size();

        MPI_Sendrecv(host_buffer_active_.data(), host_buffer_active_.size(), mpi::type_cast<typename host_buffer_type::value_type>(), destination, 0,
                     host_buffer_inactive_.data(), host_buffer_inactive_.size(), mpi::type_cast<typename host_buffer_type::value_type>(), source, 0,
                     comm_.get(), MPI_STATUS_IGNORE);

        using std::swap;
        swap(host_buffer_active_, host_buffer_inactive_);
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
