#ifndef DISTRIBUTED_K_NEAREST_NEIGHBORS_USING_LOCALITY_SENSITIVE_HASHING_AND_SYCL_MIXED_HASH_FUNCTIONS_HPP
#define DISTRIBUTED_K_NEAREST_NEIGHBORS_USING_LOCALITY_SENSITIVE_HASHING_AND_SYCL_MIXED_HASH_FUNCTIONS_HPP

#include <sycl_lsh/data.hpp>
#include <sycl_lsh/detail/assert.hpp>
#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/get_linear_id.hpp>
#include <sycl_lsh/detail/lsh_hash.hpp>
#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/device_selector.hpp>
#include <sycl_lsh/memory_layout.hpp>
#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/logger.hpp>
#include <sycl_lsh/mpi/sort.hpp>
#include <sycl_lsh/mpi/timer.hpp>
#include <sycl_lsh/options.hpp>

#include <mpi.h>

#include <random>
#include <vector>

namespace sycl_lsh {

    // SYCL kernel name needed to silence ComputeCpp warning
    class kernel_cut_off_points_unsorted;

    // forward declare class
    template <memory_layout layout, typename Options, typename Data>
    class mixed_hash_functions; // TODO 2020-10-28 16:42 marcel: name



    template <memory_layout layout, typename Options, typename Data>
    struct get_linear_id<mixed_hash_functions<layout, Options, Data>> {

        /// The used @ref sycl_lsh::options type.
        using options_type = Options;
        /// The used integral type (used for indices).
        using index_type = typename options_type::index_type;

        /// The used @ref sycl_lsh::data type.
        using data_type = Data;
        /// The used @ref sycl_lsh::data_attributes type.
        using data_attributes_type = typename data_type::data_attributes_type;

        /// The used hash functions type (mixed hash functions for this specialization).
        using hash_function_type = mixed_hash_functions<layout, Options, Data>;


        [[nodiscard]]
        index_type operator()(const index_type hash_table, const index_type hash_function, const index_type dim,
                              const options_type& opt, const data_attributes_type& attr, typename hash_function_type::buffer_part::hash_functions_t) const noexcept
        {

            if constexpr (layout == memory_layout::aos) {
                // Array of Structs
                const index_type hash_table_offset = hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1);
                return hash_table_offset + hash_function * (attr.dims + 1) + dim;
            } else {
                // Struct of Arrays
                const index_type hash_table_offset = hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1);
                return hash_table_offset + dim * opt.num_hash_functions + hash_function;
            }
        }

        [[nodiscard]]
        index_type operator()(const index_type hash_table, const index_type dim,
                              const options_type& opt, const data_attributes_type& attr, typename hash_function_type::buffer_part::hash_combine_t) const noexcept
        {
            // no difference between AoS and SoA
            const index_type hash_table_offset = hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1);
            const index_type hash_combine_offset = hash_table_offset + opt.num_hash_functions * (attr.dims + 1);
            return hash_combine_offset + dim;
        }

        [[nodiscard]]
        index_type operator()(const index_type hash_table, const index_type dim,
                              const options_type& opt, const data_attributes_type& attr, typename hash_function_type::buffer_part::cut_off_points_t) const noexcept
        {
            // no difference between AoS and SoA
            const index_type hash_table_offset = hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1);
            const index_type hash_combine_offset = hash_table_offset + opt.num_hash_functions * (attr.dims + 1);
            const index_type cut_off_points_offset = hash_combine_offset + opt.num_hash_functions;
            return cut_off_points_offset + dim;
        }

    };


    template <memory_layout layout, typename Options, typename Data>
    struct lsh_hash<mixed_hash_functions<layout, Options, Data>> {

        /// The used @ref sycl_lsh::options type.
        using options_type = Options;
        /// The used floating point type (used for the data points and hash functions).
        using real_type = typename options_type::real_type;
        /// The used integral type (used for indices).
        using index_type = typename options_type::index_type;
        /// The used unsigned type (used for the calculated hash value).
        using hash_value_type = typename options_type::hash_value_type;

        /// The used @ref sycl_lsh::data type.
        using data_type = Data;
        /// The used @ref sycl_lsh::data_attributes type.
        using data_attributes_type = typename data_type::data_attributes_type;

        /// The used hash functions type (mixed hash functions for this specialization).
        using hash_function_type = mixed_hash_functions<layout, Options, Data>;


        template <typename AccData, typename AccHashFunctions>
        [[nodiscard]]
        hash_value_type operator()(const index_type hash_table, const index_type point,
                                   AccData& acc_data, AccHashFunctions& acc_hash_functions,
                                   const options_type& opt, const data_attributes_type& attr) const
        {
            // get indexing functions
            const get_linear_id<hash_function_type> get_linear_id_hash_function{};
            const get_linear_id<data_type> get_linear_id_data{};

            real_type value = 0.0;
            for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                // calculate hash value using random projections
                real_type hash = acc_hash_functions[get_linear_id_hash_function(hash_table, hash_function, attr.dims, opt, attr, hash_function_type::buffer_part::hash_functions)];
                for (index_type dim = 0; dim < attr.dims; ++dim) {
                    hash += acc_data[get_linear_id_data(point, dim, attr)]
                            * acc_hash_functions[get_linear_id_hash_function(hash_table, hash_function, dim, opt, attr, hash_function_type::buffer_part::hash_functions)];
                }
                // combine hash values using the entropy-based hash functions
                value += static_cast<hash_value_type>(hash / opt.w)
                         * acc_hash_functions[get_linear_id_hash_function(hash_table, hash_function, opt, attr, hash_function_type::buffer_part::hash_combine)];
            }
            // calculate final hash value using the cut-off points of the combined hash values
            hash_value_type combined_hash = 0;
            for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
                combined_hash += value > acc_hash_functions[get_linear_id_hash_function(hash_table, cop, opt, attr, hash_function_type::buffer_part::cut_off_points)];
            }
            return combined_hash % opt.hash_table_size;
        }
    };



    template <memory_layout layout, typename Options, typename Data>
    class mixed_hash_functions : private detail::hash_functions_base {
        // ---------------------------------------------------------------------------------------------------------- //
        //                                      template parameter sanity checks                                      //
        // ---------------------------------------------------------------------------------------------------------- //
        static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must be a sycl_lsh::options type!");
        static_assert(std::is_base_of_v<detail::data_base, Data>, "The third template parameter must be a sycl_lsh::data type!");
    public:
        struct buffer_part {
            constexpr static struct hash_functions_t{ } hash_functions{};
            constexpr static struct hash_combine_t{ } hash_combine{};
            constexpr static struct cut_off_points_t{ } cut_off_points{};
        };

        // ---------------------------------------------------------------------------------------------------------- //
        //                                                type aliases                                                //
        // ---------------------------------------------------------------------------------------------------------- //
        /// The type of the @ref sycl_lsh::options object.
        using options_type = Options;
        /// The used floating point type for the hash functions.
        using real_type = typename options_type::real_type;
        /// The used integral type for indices.
        using index_type = typename options_type::index_type;
        /// The used unsigned type for the hash values.
        using hash_value_type = typename options_type::hash_value_type;

        /// The type of the @ref sycl_lsh::data object.
        using data_type = Data;
        /// The type of the @ref sycl_lsh::data_attributes object.
        using data_attributes_type = typename data_type::data_attributes_type;

        /// The type of the device buffer used by SYCL.
        using device_buffer_type = sycl::buffer<real_type, 1>;


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                constructor                                                 //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Construct a new @ref sycl_lsh::entropy_based object representing the hash functions used in the LSH algorithm.
         * @param[in] opt the used @ref sycl_lsh::options
         * @param[in] data the used @ref sycl_lsh::data
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         * @param[in] logger the used @ref sycl_lsh::mpi::logger
         */
        mixed_hash_functions(const options_type& opt, data_type& data, const mpi::communicator& comm, const mpi::logger& logger);


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
         * @brief Returns the device buffer used in the SYCL kernels.
         * @return the device buffer (`[[nodiscard]]`)
         */
        [[nodiscard]]
        device_buffer_type& get_device_buffer() noexcept { return device_buffer_; }

    private:
        device_buffer_type device_buffer_;
    };


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options, typename Data>
    mixed_hash_functions<layout, Options, Data>::mixed_hash_functions(const options_type& opt, data_type& data,
                                                        const mpi::communicator& comm, const mpi::logger& logger)
            : device_buffer_(opt.num_hash_tables * opt.num_hash_functions * (data.get_attributes().dims + 1) + // random projections as hash functions
                             opt.num_hash_tables * (opt.num_hash_functions + opt.num_cut_off_points - 1))      // entropy-based as hash combine
    {
        mpi::timer t(comm);

        const data_attributes_type attr = data.get_attributes();

        std::vector<real_type> host_buffer(device_buffer_.get_count());
        const get_linear_id<mixed_hash_functions<layout, options_type, data_type>> get_linear_id_functor{};

        //
        // CREATE RANDOM PROJECTIONS HASH FUNCTIONS
        //

        // create hash pool only on MPI master rank
        if (comm.master_rank()) {
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
                    hash_pool[hash_function * (attr.dims + 1) + dim] = std::abs(rnd_normal_pool_dist(rnd_normal_pool_gen));
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
                        host_buffer[get_linear_id_functor(hash_table, hash_function, dim, opt, attr, buffer_part::hash_functions)]
                                = hash_pool[pool_hash_function * (attr.dims + 1) + dim];
                    }
                }
            }
        }


        //
        // CREATE ENTROPY-BASED HASH FUNCTIONS
        //

        if (comm.master_rank()) {
            // create random generator
            #if SYCL_LSH_DEBUG
                // don't seed random engine in debug mode
                std::mt19937 rnd_normal_pool_gen;
            #else
                // seed random engine outside debug mode
                std::random_device rnd_pool_device;
                std::mt19937 rnd_normal_pool_gen(rnd_pool_device());
            #endif
            std::normal_distribution<real_type> rnd_normal_dist;

            // fill hash functions
            for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
                for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                    host_buffer[get_linear_id_functor(hash_table, hash_function, opt, attr, buffer_part::hash_combine)]
                            = rnd_normal_dist(rnd_normal_pool_gen);
                }
            }
        }

        // broadcast random projections hash functions to other MPI ranks
        MPI_Bcast(host_buffer.data(), host_buffer.size(), mpi::type_cast<real_type>(), 0, comm.get());


        // calculate cut-off points
        {
            sycl::queue queue(device_selector{comm}, sycl::async_handler(&sycl_exception_handler));
            sycl::buffer<real_type, 1> hash_functions_buffer(host_buffer.data(), host_buffer.size());

            std::vector<real_type> hash_values(attr.rank_size);
            for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
                {
                    sycl::buffer<real_type, 1> hash_values_buffer(hash_values.data(), hash_values.size());
                    queue.submit([&](sycl::handler& cgh) {
                        auto acc_data = data.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
                        auto acc_hash_functions = hash_functions_buffer.template get_access<sycl::access::mode::read>(cgh);
                        auto acc_hash_values = hash_values_buffer.template get_access<sycl::access::mode::discard_write>(cgh);

                        const options_type options = opt;
                        const data_attributes_type attributes = attr;
                        const get_linear_id<data_type> get_linear_id_data{};
                        const get_linear_id<mixed_hash_functions<layout, options_type, data_type>> get_linear_id_hash_functions{};

                        cgh.parallel_for<kernel_cut_off_points_unsorted>(sycl::range<>(attr.rank_size), [=](sycl::item<> item) {
                            const index_type idx = item.get_linear_id();

                            real_type value = 0.0;
                            for (index_type hash_function = 0; hash_function < options.num_hash_functions; ++hash_function) {
                                real_type hash = acc_hash_functions[get_linear_id_hash_functions(hash_table, hash_function, attributes.dims, options, attributes, buffer_part::hash_functions)];
                                for (index_type dim = 0; dim < attributes.dims; ++dim) {
                                    hash += acc_data[get_linear_id_data(idx, dim, attributes)]
                                            * acc_hash_functions[get_linear_id_hash_functions(hash_table, hash_function, dim, options, attributes, buffer_part::hash_functions)];
                                }
                                value += static_cast<hash_value_type>(hash / options.w)
                                         * acc_hash_functions[get_linear_id_hash_functions(hash_table, hash_function, options, attributes, buffer_part::hash_combine)];
                            }
                            acc_hash_values[idx] = value;
                        });
                    });
                }

                // sort hash_values vector in a distributed fashion
                mpi::sort(hash_values, comm);

                std::vector<real_type> cut_off_points(opt.num_cut_off_points - 1, 0.0);

                // calculate cut-off points indices
                std::vector<index_type> cut_off_points_idx(cut_off_points.size());
                const index_type jump = (attr.rank_size * comm.size()) / opt.num_cut_off_points;
                for (index_type cop = 0; cop < cut_off_points_idx.size(); ++cop) {
                    cut_off_points_idx[cop] = (cop + 1) * jump;
                }

                // fill cut-off points which are located on the current MPI rank
                for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
                    // check if index belongs to current MPI rank
                    if (cut_off_points_idx[cop] >= attr.rank_size * comm.rank() && cut_off_points_idx[cop] < attr.rank_size * (comm.rank() + 1)) {
                        cut_off_points[cop] = hash_values[cut_off_points_idx[cop] % attr.rank_size];
                    }
                }

                // combine to final cut-off points on all MPI ranks
                MPI_Allreduce(MPI_IN_PLACE, cut_off_points.data(), cut_off_points.size(), mpi::type_cast<real_type>(), MPI_SUM, comm.get());

                // copy current cut-off points to hash functions
                const get_linear_id<mixed_hash_functions<layout, options_type, data_type>> get_linear_id_functor;
                for (index_type cop = 0; cop < cut_off_points.size(); ++cop) {
                    host_buffer[get_linear_id_functor(hash_table, cop, opt, attr, buffer_part::cut_off_points)] = cut_off_points[cop];
                }
            }
        }

        // broadcast hash function to other MPI ranks
        MPI_Bcast(host_buffer.data(), host_buffer.size(), mpi::type_cast<real_type>(), 0, comm.get());


        // copy data to device buffer
        auto acc = device_buffer_.template get_access<sycl::access::mode::discard_write>();
        for (index_type i = 0; i < acc.get_count(); ++i) {
            acc[i] = host_buffer[i];
        }

        logger.log("Created 'mixed_hash_functions' hash functions in {}.\n", t.elapsed());
    }

}

#endif // DISTRIBUTED_K_NEAREST_NEIGHBORS_USING_LOCALITY_SENSITIVE_HASHING_AND_SYCL_MIXED_HASH_FUNCTIONS_HPP
