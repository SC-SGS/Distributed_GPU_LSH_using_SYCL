/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-08
 *
 * @brief Implements the random projections hash function as the used LSH hash functions.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_RANDOM_PROJECTIONS_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_RANDOM_PROJECTIONS_HPP

#include <sycl_lsh/detail/assert.hpp>
#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/get_linear_id.hpp>
#include <sycl_lsh/detail/lsh_hash.hpp>
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


    /**
     * @brief Specialization of the @ref sycl_lsh::get_linear_id class for the @ref sycl_lsh::random_projections class to convert a
     *        multi-dimensional index to an one-dimensional one.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the @ref sycl_lsh::options type
     * @tparam Data the @ref sycl_lsh::data type
     */
    template <memory_layout layout, typename Options, typename Data>
    struct get_linear_id<random_projections<layout, Options, Data>> {

        /// The used @ref sycl_lsh::options type.
        using options_type = Options;
        /// The used integral type (used for indices).
        using index_type = typename options_type::index_type;

        /// The used @ref sycl_lsh::data type.
        using data_type = Data;
        /// The used @ref sycl_lsh::data_attributes type.
        using data_attributes_type = typename data_type::data_attributes_type;

        /**
         * @brief Convert the multi-dimensional index to an one-dimensional index.
         * @param[in] hash_table the requested hash table
         * @param[in] hash_function the requested hash function
         * @param[in] dim the requested dimension of @p hash_function
         * @param[in] opt the used @ref sycl_lsh::options
         * @param[in] attr the attributes of the used data set
         * @return the one-dimensional index (`[[nodiscard]]`)
         *
         * @pre @p hash_table must be in the range `[0, number of hash tables)` (currently disabled).
         * @pre @p hash_function must be in the range `[0, number of hash functions)` (currently disabled).
         * @pre @p dim must be in the range `[0, number of dimensions per data point + 1)` (currently disabled).
         */
        [[nodiscard]]
        index_type operator()(const index_type hash_table, const index_type hash_function, const index_type dim,
                              const options_type& opt, const data_attributes_type& attr) const noexcept
        {
//            SYCL_LSH_DEBUG_ASSERT(0 <= hash_table && hash_table < opt.num_hash_tables, "Out-of-bounce access for hash table!\n");
//            SYCL_LSH_DEBUG_ASSERT(0 <= hash_function && hash_function < opt.hash_pool_size, "Out-of-bounce access for hash function!\n");
//            SYCL_LSH_DEBUG_ASSERT(0 <= dim && dim < attr.dims, "Out-of-bounce access for dimension!\n");

            if constexpr (layout == memory_layout::aos) {
                // Array of Structs
                return hash_table * opt.num_hash_functions * (attr.dims + 1) + hash_function * (attr.dims + 1) + dim;
            } else {
                // Struct of Arrays
                return hash_table * opt.num_hash_functions * (attr.dims + 1) + dim * opt.num_hash_functions + hash_function;
            }
        }

    };

    /**
     * @brief Specialization of the @ref sycl_lsh::lsh_hash class for the @ref sycl_lsh::random_projections class to calculate the
     *        hash value.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the @ref sycl_lsh::options type
     * @tparam Data the @ref sycl_lsh::data type
     */
    template <memory_layout layout, typename Options, typename Data>
    struct lsh_hash<random_projections<layout, Options, Data>> {

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

        /// The used hash functions type (random projections for this specialization).
        using hash_function_type = random_projections<layout, Options, Data>;

        /**
         * @brief Calculates the hash value of the data point @p point in hash table @p hash_tables using random projections.
         * @tparam AccData the type of the data set `sycl::accessor`
         * @tparam AccHashFunctions the type of the hash functions `sycl::accessor`
         * @param[in] hash_table the provided hash table
         * @param[in] point the provided data point
         * @param[in] acc_data the data set `sycl::accessor`
         * @param[in] acc_hash_functions the hash functions `sycl::accessor`
         * @param[in] opt the used @ref sycl_lsh::options
         * @param[in] attr the used @ref sycl_lsh::data_attributes
         * @return the calculated hash value using random projections (`[[nodiscard]]`)
         *
         * @pre @p hash_table must be in the range `[0, number of hash tables)` (currently disabled).
         * @pre @p hash_function must be in the range `[0, number of hash functions)` (currently disabled).
         */
        template <typename AccData, typename AccHashFunctions>
        [[nodiscard]]
        hash_value_type operator()(const index_type hash_table, const index_type point,
                                   AccData& acc_data, AccHashFunctions& acc_hash_functions,
                                   const options_type& opt, const data_attributes_type& attr) const

        {
//            SYCL_LSH_DEBUG_ASSERT(0 <= hash_table && hash_table < opt.num_hash_tables, "Out-of-bounce access for hash tables!\n");
//            SYCL_LSH_DEBUG_ASSERT(0 <= point && point < attr.rank_size, "Out-of-bounce access for data point!");

            // get indexing functions
            const get_linear_id<hash_function_type> get_linear_id_hash_function{};
            const get_linear_id<data_type> get_linear_id_data{};

            hash_value_type combined_hash = opt.num_hash_functions;
            for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                // calculate hash for current hash function
                real_type hash = acc_hash_functions[get_linear_id_hash_function(hash_table, hash_function, attr.dims, opt, attr)];
                for (index_type dim = 0; dim < attr.dims; ++dim) {
                    hash += acc_data[get_linear_id_data(point, dim, attr)]
                            * acc_hash_functions[get_linear_id_hash_function(hash_table, hash_function, dim, opt, attr)];
                }
                // combine hashes
                combined_hash ^= static_cast<hash_value_type>(hash / opt.w)
                                + static_cast<hash_value_type>(0x9e3779b9)
                                + (combined_hash << static_cast<hash_value_type>(6))
                                + (combined_hash >> static_cast<hash_value_type>(2));
            }
            return combined_hash % opt.hash_table_size;
        }

    };


    /**
     * @brief Class which represents the random projections hash functions used in the LSH algorithm.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     * @tparam Data the used @ref sycl_lsh::data type
     */
    template <memory_layout layout, typename Options, typename Data>
    class random_projections : private detail::hash_functions_base {
        // ---------------------------------------------------------------------------------------------------------- //
        //                                      template parameter sanity checks                                      //
        // ---------------------------------------------------------------------------------------------------------- //
        static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must be a sycl_lsh::options type!");
        static_assert(std::is_base_of_v<detail::data_base, Data>, "The third template parameter must be a sycl_lsh::data type!");
    public:
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
         * @brief Construct a new @ref sycl_lsh::random_projections object representing the hash functions used in the LSH algorithm.
         * @param[in] opt the used @ref sycl_lsh::options
         * @param[in] data the used @ref sycl_lsh::data
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         * @param[in] logger the used @ref sycl_lsh::mpi::logger
         */
        random_projections(const options_type& opt, const data_type& data, const mpi::communicator& comm, const mpi::logger& logger);


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
    random_projections<layout, Options, Data>::random_projections(const options_type& opt, const data_type& data,
                                                                  const mpi::communicator& comm, const mpi::logger& logger)
            : device_buffer_(opt.num_hash_tables * opt.num_hash_functions * (data.get_attributes().dims + 1))
    {
        mpi::timer t(comm);

        const data_attributes_type& attr = data.get_attributes();

        std::vector<real_type> host_buffer(device_buffer_.get_count());

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
                    // TODO 2020-10-02 12:47 marcel: abs?
                    hash_pool[hash_function * (attr.dims + 1) + dim] = rnd_normal_pool_dist(rnd_normal_pool_gen);
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

            const get_linear_id<random_projections<layout, options_type, data_type>> get_linear_id_functor;

            for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
                for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                    const index_type pool_hash_function = rnd_uniform_dist(rnd_uniform_gen);
                    for (index_type dim = 0; dim <= attr.dims; ++dim) {
                        host_buffer[get_linear_id_functor(hash_table, hash_function, dim, opt, attr)]
                                = hash_pool[pool_hash_function * (attr.dims + 1) + dim];
                    }
                }
            }
        }

        // broadcast hash functions to other MPI ranks
        MPI_Bcast(host_buffer.data(), host_buffer.size(), mpi::type_cast<real_type>(), 0, comm.get());

        // copy data to device buffer
        auto acc = device_buffer_.template get_access<sycl::access::mode::discard_write>();
        for (index_type i = 0; i < acc.get_count(); ++i) {
            acc[i] = host_buffer[i];
        }

        logger.log("Created 'random_projections' hash functions in {}.\n", t.elapsed());
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_RANDOM_PROJECTIONS_HPP
