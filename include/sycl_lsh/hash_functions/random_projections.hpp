/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-30
 *
 * @brief Implements the random projections hash function as the used LSH hash functions.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_RANDOM_PROJECTIONS_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_RANDOM_PROJECTIONS_HPP

#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/data.hpp>
#include <sycl_lsh/memory_layout.hpp>
#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/logger.hpp>
#include <sycl_lsh/mpi/timer.hpp>
#include <sycl_lsh/options.hpp>

#include <mpi.h>

#include <random>
#include <vector>

namespace sycl_lsh {

    // forward declare random projections class
    template <memory_layout layout, typename Options, typename Data>
    class random_projections;

    
    template <memory_layout layout, typename Options, typename Data>
    [[nodiscard]]
    inline auto make_random_projection_hash_function(const Options& opt, const Data& data, const mpi::communicator& comm, const mpi::logger& logger) {
        return random_projections<layout, Options, Data>(opt, data, comm, logger);
    }

    // TODO 2020-09-30 14:02 marcel: get linear id: template parameter? other hash function???
    template <memory_layout layout, typename index_type, typename Options>
    [[nodiscard]]
    constexpr index_type get_linear_id__hash_function(const index_type hash_table, const index_type hash_function, const index_type dim,
                                                      const Options& opt, const data_attributes<layout, index_type>& data_attr) noexcept
    {
        SYCL_LSH_DEBUG_ASSERT(0 <= hash_table && hash_table < opt.num_hash_tables, "Out-of-bounce access for hash table!\n");
        SYCL_LSH_DEBUG_ASSERT(0 <= hash_function && hash_function < opt.hash_pool_size, "Out-of-bounce access for hash function!\n");
        SYCL_LSH_DEBUG_ASSERT(0 <= dim && dim < data_attr.dims, "Out-of-bounce access for dimension!\n");

        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return hash_table * opt.num_hash_functions * (data_attr.dims + 1) + hash_function * (data_attr.dims + 1) + dim;
        } else {
            // Struct of Arrays
            return hash_table * opt.num_hash_functions * (data_attr.dims + 1) + dim * opt.num_hash_functions + hash_function;
        }
    }


    template <memory_layout layout, typename Options, typename Data>
    class random_projections : detail::hash_functions_base {
        // ---------------------------------------------------------------------------------------------------------- //
        //                                      template parameter sanity checks                                      //
        // ---------------------------------------------------------------------------------------------------------- //
        static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must be a sycl_lsh::options type!");
        static_assert(std::is_base_of_v<detail::data_base, Data>, "The third template parameter must be a sycl_lsh::data type!");
    public:
        using options_type = Options;
        using data_type = Data;
        using index_type = typename options_type::index_type;
        using real_type = typename options_type::real_type;
        using hash_value_type = typename options_type::hash_value_type;

        using device_buffer_type = sycl::buffer<real_type, 1>;


        [[nodiscard]]
        constexpr memory_layout get_memory_layout() const noexcept { return layout; }
        [[nodiscard]]
        const options_type& get_options() const noexcept { return options_; }
        [[nodiscard]]
        const data_type& get_data() const noexcept { return data_; }

        [[nodiscard]]
        device_buffer_type& get_device_buffer() const noexcept { return device_buffer_; }

    private:
        // befriend factory function
        friend auto make_random_projection_hash_function<layout, Options, Data>(const options_type&, const data_type&, const mpi::communicator&, const mpi::logger&);


        random_projections(const options_type& opt, const data_type& data, const mpi::communicator& comm, const mpi::logger& logger)
            : options_(opt), data_(data), comm_(comm), logger_(logger),
              device_buffer_(opt.num_hash_tables * opt.num_hash_functions * (data.get_attributes().dims + 1))
        {
            mpi::timer t(comm_);

            const auto& attr = data.get_attributes();

            std::vector<real_type> host_buffer(device_buffer_.get_count());

            // create hash pool only on MPI master rank
            if (comm_.master_rank()) {
                // create random generators
                #if SYCL_LSH_DEBUG
                    // don't seed random engine in debug mode
                    std::mt19937 rnd_normal_pool_gen;
                    std::mt19937 rnd_uniform_pool_gen;
                #else
                    // seed random engine outside debug mode
                    std::random_device rnd_pool_device;
                    std::mt19937 rnd_normal_pool_gen(rnd_pool_device());
                    std::mt19937 rnd_uniform_pool_gen(rnd_pool_device());
                #endif
                std::normal_distribution<real_type> rnd_normal_pool_dist;
                std::uniform_real_distribution<real_type> rnd_uniform_pool_dist(0, opt.w);

                // fill hash pool
                std::vector<real_type> hash_pool(opt.hash_pool_size * (attr.dims + 1));
                for (index_type hash_function = 0; hash_function < opt.hash_pool_size; ++hash_function) {
                    for (index_type dim = 0; dim < attr.dims; ++dim) {
                        hash_pool[hash_function * (attr.dims + 1) + dim] = rnd_normal_pool_dist(rnd_uniform_pool_gen);
                    }
                    hash_pool[hash_function * (attr.dims + 1) + attr.dims] = rnd_uniform_pool_dist(rnd_uniform_pool_gen);
                }

                // select actual hash functions
                #if SYCL_LSH_DEBUG
                    // don't seed random engine in debug mode
                    std::mt19937 rnd_uniform_gen;
                #else
                    // seed random engine outside debug mode
                    std::random_device rnd_device;
                    std::mt19937 rnd_uniform_gen(rnd_device());
                #endif
                std::uniform_int_distribution<index_type> rnd_uniform_dist(0, opt.hash_pool_size - 1);

                for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
                    for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                        const index_type pool_hash_function = rnd_uniform_dist(rnd_uniform_gen);
                        for (index_type dim = 0; dim <= attr.dims; ++dim) {
                            host_buffer[get_linear_id__hash_function(hash_table, hash_function, dim, opt, attr)]
                                = hash_pool[pool_hash_function * (attr.dims + 1) * dim];
                        }
                    }
                }
            }

            // broadcast hash functions to other MPI ranks
            MPI_Bcast(host_buffer.data(), host_buffer.size(), mpi::type_cast<real_type>(), 0, comm_.get());

            // copy data to device buffer
            auto acc = device_buffer_.template get_access<sycl::access::mode::discard_write>();
            for (index_type i = 0; i < acc.get_count(); ++i) {
                acc[i] = host_buffer[i];
            }

            logger_.log("Created hash functions in {}.\n", t.elapsed());
        }


        const options_type& options_;
        const data_type& data_;
        const mpi::communicator& comm_;
        const mpi::logger& logger_;

        device_buffer_type device_buffer_;
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_RANDOM_PROJECTIONS_HPP
