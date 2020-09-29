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

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace sycl_lsh {

    template <memory_layout layout, typename Options>
    class data;

    template <memory_layout layout, typename Options>
    [[nodiscard]]
    inline auto make_data(const argv_parser& parser, const Options& opt, const mpi::communicator& comm, const mpi::logger& logger) {
        using real_type = typename Options::real_type;
        auto file_parser = std::make_unique<mpi::binary_parser<Options, real_type>>(parser.argv_as<std::string>("data_file"), comm, logger);
        return data<layout, Options>(*file_parser, opt, comm, logger);
    }


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


    template <memory_layout layout, typename Options>
    class data : public detail::data_base {
        // ---------------------------------------------------------------------------------------------------------- //
        //                                      template parameter sanity checks                                      //
        // ---------------------------------------------------------------------------------------------------------- //
        static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must be a sycl_lsh::options type!");
    public:
        using index_type = typename Options::index_type;
        using real_type = typename Options::real_type;
        using options_type = Options;
        using data_attributes_type = data_attributes<layout, index_type>;

        using gpu_buffer_type = sycl::buffer<real_type, 1>;
        using host_buffer_type = std::vector<real_type>;


        [[nodiscard]]
        constexpr memory_layout get_memory_layout() const noexcept { return layout; }
        [[nodiscard]]
        const options_type& get_options() const noexcept { return options_; }
        [[nodiscard]]
        const data_attributes_type& get_data_attributes() const noexcept { return data_attributes_; }

        [[nodiscard]]
        gpu_buffer_type get_gpu_buffer() noexcept { return gpu_buffer_; }
        [[nodiscard]]
        host_buffer_type get_host_buffer() noexcept { return host_buffer_active_; }

    private:
        friend auto make_data<layout, Options>(const argv_parser&, const Options&, const mpi::communicator&, const mpi::logger&);
        

        data(const mpi::file_parser<options_type, real_type>& parser, const options_type& opt, const mpi::communicator& comm, const mpi::logger& logger)
            : options_(opt), comm_(comm), logger_(logger),
              data_attributes_(parser.parse_total_size(), parser.parse_rank_size(), parser.parse_dims()),
              gpu_buffer_(data_attributes_.rank_size * data_attributes_.dims),
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
            auto acc = gpu_buffer_.template get_access<sycl::access::mode::discard_write>();
            for (index_type i = 0; i < acc.get_count(); ++i) {
                acc[i] = host_buffer_active_[i];
            }

            logger_.log("Created data object in {}.\n", t.elapsed());
        }

        const options_type& options_;
        const mpi::communicator& comm_;
        const mpi::logger& logger_;

        const data_attributes_type data_attributes_;

        gpu_buffer_type gpu_buffer_;
        host_buffer_type host_buffer_active_;
        host_buffer_type host_buffer_inactive_;
    };


    template <memory_layout layout, typename Options>
    std::ostream& operator<<(std::ostream& out, const data<layout, Options>& d) {
        return out << d.get_data_attributes();
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
