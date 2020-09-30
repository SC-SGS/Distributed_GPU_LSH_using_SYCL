/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-29
 *
 * @brief Implements the @ref sycl_lsh::data class representing the used data set.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP

#include <sycl_lsh/argv_parser.hpp>
#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/memory_layout.hpp>
#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/logger.hpp>
#include <sycl_lsh/options.hpp>
#include <sycl_lsh/data_attributes.hpp>
#include <sycl_lsh/mpi/file_parser/file_parser.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

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
    inline auto make_data(const argv_parser& parser, const Options& opt, const mpi::communicator& comm, const mpi::logger& logger) {
        using real_type = typename Options::real_type;
        auto file_parser = mpi::make_file_parser<real_type, Options>(parser, comm, logger);
        return data<layout, Options>(*file_parser, opt, comm, logger);
    }

    /**
     * @brief Convert the multi-dimensional index to an one-dimensional index.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam index_type an integral type (used for indices)
     * @param[in] point the requested data point
     * @param[in] dim the requested dimension of @p point
     * @param[in] data_attr the attributes of the used data set
     * @return the one-dimensional index (`[[nodiscard]]`)
     *
     * @pre @p point must be in the range `[0, number of data points on the current MPI rank)`.
     * @pre @p dim must be in the range `[0, number of dimensions per data point)`
     */
    template <memory_layout layout, typename index_type>
    [[nodiscard]]
    constexpr index_type get_linear_id__data(const index_type point, const index_type dim, const data_attributes<layout, index_type>& data_attr) noexcept {
        SYCL_LSH_DEBUG_ASSERT(0 <= point && point < data_attr.rank_size, "Out-of-bounce access for data point!\n");
        SYCL_LSH_DEBUG_ASSERT(0 <= dim && dim < data_attr.dims, "Out-of-bounce access for dimension!\n");

        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return dim + point * data_attr.dims;
        } else {
            // Struct of Arrays
            return point + dim * data_attr.rank_size;
        }
    }


    /**
     * @brief Class which represents the used data set.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     */
    template <memory_layout layout, typename Options>
    class data : public detail::data_base {
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
        //                                                   getter                                                   //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Returns the specified @ref sycl_lsh::memory_layout type.
         * @return the @ref sycl_lsh::memory_layout type (`[[nodiscard]]`)
         */
        [[nodiscard]]
        constexpr memory_layout get_memory_layout() const noexcept { return layout; }
        /**
         * @brief Return the @ref sycl_lsh::options object used to control the behavior of the used algorithms.
         * @return the @ref sycl_lsh::options (`[[nodiscard]]`)
         */
        [[nodiscard]]
        const options_type& get_options() const noexcept { return options_; }
        /**
         * @brief Return the @ref sycl_lsh::data_attributes object representing the attributes of the used data set.
         * @return the @ref sycl_lsh::data_attributes (`[[nodiscard]]`)
         */
        [[nodiscard]]
        const data_attributes_type& get_attributes() const noexcept { return data_attributes_; }

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
        data(const mpi::file_parser<options_type, real_type>& parser, const options_type& opt, const mpi::communicator& comm, const mpi::logger& logger);


        const options_type& options_;
        const mpi::communicator& comm_;
        const mpi::logger& logger_;

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
                                const Options& opt,
                                const mpi::communicator& comm,
                                const mpi::logger& logger)
            : options_(opt), comm_(comm), logger_(logger),
              data_attributes_(parser.parse_total_size(), parser.parse_rank_size(), parser.parse_dims()),
              device_buffer_(data_attributes_.rank_size * data_attributes_.dims),
              host_buffer_active_(data_attributes_.rank_size * data_attributes_.dims),
              host_buffer_inactive_(data_attributes_.rank_size * data_attributes_.dims)
    {
        mpi::timer t(comm_);

        // parse content
        parser.parse_content(host_buffer_active_.data());

        // change memory layout from aos to soa if requested
        if constexpr (layout == memory_layout::soa) {
            data_attributes<memory_layout::aos, index_type> parsed_data_attributes(data_attributes_);

            for (index_type point = 0; point < data_attributes_.rank_size; ++point) {
                for (index_type dim = 0; dim < data_attributes_.dims; ++dim) {
                    host_buffer_inactive_[get_linear_id__data(point, dim, data_attributes_)]
                            = host_buffer_active_[get_linear_id__data(point, dim, parsed_data_attributes)];
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

        logger_.log("Created data object in {}.\n", t.elapsed());
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
