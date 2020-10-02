/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-02
 *
 * @brief Implements the entropy based hash function as the used LSH hash functions.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ENTROPY_BASED_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ENTROPY_BASED_HPP

#include <sycl_lsh/detail/assert.hpp>
#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/get_linear_id.hpp>
#include <sycl_lsh/detail/lsh_hash.hpp>
#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/data.hpp>
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
    class cut_off_points_unsorted;

    // forward declare entropy based class
    template <memory_layout layout, typename Options, typename Data>
    class entropy_based;

    /**
     * @brief Factory function for the @ref sycl_lsh::entropy_based hash functions class.
     * @details Used to be able to automatically deduce the @ref sycl_lsh::options and @ref sycl_lsh::data types.
     * @tparam layout the used @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     * @tparam Data the used @ref sycl_lsh::data type
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] data the used @ref sycl_lsh::data
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger hte used @ref sycl_lsh::mpi::logger
     * @return the @ref sycl_lsh::entropy_based hash functions used in the LSH algorithm (`[[nodiscard]]`)
     */
    template <memory_layout layout, typename Options, typename Data>
    [[nodiscard]]
    inline auto make_entropy_based_hash_functions(const Options& opt, Data& data, const mpi::communicator& comm, const mpi::logger& logger) {
        return entropy_based<layout, Options, Data>(opt, data, comm, logger);
    }

    /**
     * @brief Specialization of the @ref sycl_lsh::get_linear_id class for the @ref sycl_lsh::entropy_based class to convert a
     *        multi-dimensional index to an one-dimensional one.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the @ref sycl_lsh::options type
     * @tparam Data the @ref sycl_lsh::data type
     */
    template <memory_layout layout, typename Options, typename Data>
    struct get_linear_id<entropy_based<layout, Options, Data>> {

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
                return hash_table * opt.num_hash_functions * (attr.dims + opt.num_cut_off_points - 1)
                        + hash_function * (attr.dims + opt.num_cut_off_points - 1)
                        + dim;
            } else {
                // Struct of Arrays
                return hash_table * opt.num_hash_functions * (attr.dims + opt.num_cut_off_points - 1)
                        + dim * opt.num_hash_functions
                        + hash_function;
            }
        }
        
    };

    /**
     * @brief Specialization of the @ref sycl_lsh::lsh_hash class for the @ref sycl_lsh::entropy_based class to calculate the
     *        hash value.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the @ref sycl_lsh::options type
     * @tparam Data the @ref sycl_lsh::data type
     */
    template <memory_layout layout, typename Options, typename Data>
    struct lsh_hash<entropy_based<layout, Options, Data>> {

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

        /// The used hash functions type (entropy based for this specialization).
        using hash_function_type = entropy_based<layout, Options, Data>;

        /**
         * @brief Calculates the hash value of the data point @p point in hash table @p hash_tables using entropy based hash functions.
         * @tparam AccData the type of the data set `sycl::accessor`
         * @tparam AccHashFunctions the type of the hash functions `sycl::accessor`
         * @param[in] hash_table the provided hash table
         * @param[in] point the provided data point
         * @param[in] acc_data the data set `sycl::accessor`
         * @param[in] acc_hash_functions the hash functions `sycl::accessor`
         * @param[in] opt the used @ref sycl_lsh::options
         * @param[in] attr the used @ref sycl_lsh::data_attributes
         * @return the calculated hash value using entropy based hash functions (`[[nodiscard]]`)
         *
         * @pre @p hash_table must be in the range `[0, number of hash tables)` (currently disabled).
         * @pre @p hash_function must be in the range `[0, number of hash functions)` (currently disabled).
         */
        template <typename AccData, typename AccHashFunctions>
        [[nodiscard]]
        hash_value_type  operator()(const index_type hash_table, const index_type point,
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
                // calculate dot product for current hash function
                real_type hash = 0.0;
                for (index_type dim = 0; dim < attr.dims; ++dim) {
                    hash += acc_data[get_linear_id_data(point, dim, attr)]
                            * acc_hash_functions[get_linear_id_hash_function(hash_table, hash_function, dim, opt, attr)];
                }
                // calculate entropy hash for current hash function
                hash_value_type entropy_hash = 0;
                for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
                    entropy_hash += hash > acc_hash_functions[get_linear_id_hash_function(hash_table, hash_function, attr.dims + cop, opt, attr)];
                }
                // combine hashes
                combined_hash ^= entropy_hash
                                 + static_cast<hash_value_type>(0x9e3779b9)
                                 + (combined_hash << static_cast<hash_value_type>(6))
                                 + (combined_hash >> static_cast<hash_value_type>(2));
            }
            return combined_hash % opt.hash_table_size;
        }
    };


    /**
     * @brief Class which represents the entropy based hash functions used in the LSH algorithm.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     * @tparam Data the used @ref sycl_lsh::data type
     */
    template <memory_layout layout, typename Options, typename Data>
    class entropy_based : detail::hash_functions_base {
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
        /// The type of the @ref sycl_lsh::data object.
        using data_type = Data;
        /// The used floating point type for the hash functions.
        using real_type = typename options_type::real_type;
        /// The used integral type for indices.
        using index_type = typename options_type::index_type;
        /// The used unsigned type for the hash values.
        using hash_value_type = typename options_type::hash_value_type;

        using data_attributes_type = typename data_type::data_attributes_type;

        /// The type of the device buffer used by SYCL.
        using device_buffer_type = sycl::buffer<real_type, 1>;


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
         * @brief Returns the @ref sycl_lsh::options object used to control the behavior of the used algorithm.
         * @return the @ref sycl_lsh::options (`[[nodiscard]]`)
         */
        [[nodiscard]]
        const options_type get_options() const noexcept { return options_; }
        /**
         * @brief Returns the @ref sycl_lsh::data object representing the used data set.
         * @return the @ref sycl_lsh::data (`[[nodiscard]]`)
         */
        [[nodiscard]]
        data_type& get_data() const noexcept { return data_; }

        /**
         * @brief Returns the device buffer used in the SYCL kernels.
         * @return the device buffer (`[[nodiscard]]`)
         */
        [[nodiscard]]
        device_buffer_type& get_device_buffer() noexcept { return device_buffer_; }

    private:
        // befriend factory function
        friend auto make_entropy_based_hash_functions<layout, Options, Data>(const options_type&, data_type&, const mpi::communicator&, const mpi::logger&);

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
        entropy_based(const options_type& opt, data_type& data, const mpi::communicator& comm, const mpi::logger& logger);


        const options_type& options_;
        data_type& data_;
        const mpi::communicator& comm_;
        const mpi::logger& logger_;

        device_buffer_type device_buffer_;
    };


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options, typename Data>
    entropy_based<layout, Options, Data>::entropy_based(const Options& opt, Data& data,
                                                        const mpi::communicator& comm, const mpi::logger& logger)
            : options_(opt), data_(data), comm_(comm), logger_(logger),
              device_buffer_(opt.num_hash_tables * opt.num_hash_functions * (data.get_attributes().dims + opt.num_cut_off_points - 1))
    {
        mpi::timer t(comm_);

        const data_attributes_type attr = data.get_attributes();

        // create hash pool functions on MPI master rank and distribute to all other ranks
        std::vector<real_type> hash_functions_pool(options_.hash_pool_size * attr.dims);

        const auto get_linear_id_hash_pool = [=](const index_type hash_function, const index_type dim,
                                                 [[maybe_unused]] const options_type& opt, [[maybe_unused]] const data_attributes_type& attr)
        {
            if constexpr (layout == memory_layout::aos) {
                // Array of Structs
                return hash_function * attr.dims + dim;
            } else {
                // Struct of Arrays
                return dim * opt.hash_pool_size + hash_function;
            }
        };

        if (comm_.master_rank()) {
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
            for (index_type hash_function = 0; hash_function < options_.hash_pool_size; ++hash_function) {
                for (index_type dim = 0; dim < attr.dims; ++dim) {
                    hash_functions_pool[get_linear_id_hash_pool(hash_function, dim, options_, attr)] = rnd_normal_dist(rnd_normal_pool_gen);
                }
            }
        }

        // broadcast pool hash functions to other MPI ranks
        MPI_Bcast(hash_functions_pool.data(), hash_functions_pool.size(), mpi::type_cast<real_type>(), 0, comm_.get());

        std::vector<real_type> cut_off_points_pool(options_.hash_pool_size * (options_.num_cut_off_points - 1));

        // calculate cut-off points
        {
            // TODO 2020-10-02 13:16 marcel: change to custom selector
            sycl::queue queue(sycl::default_selector{});
            sycl::buffer<real_type, 1> hash_functions_pool_buffer(hash_functions_pool.data(), hash_functions_pool.size());

            std::vector<real_type> hash_values(attr.rank_size);
            for (index_type hash_function = 0; hash_function < options_.hash_pool_size; ++hash_function) {
                {
                    sycl::buffer<real_type, 1> hash_values_buffer(hash_values.data(), hash_values.size());
                    queue.submit([&](sycl::handler& cgh) {
                        auto acc_data = data_.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
                        auto acc_hash_functions = hash_functions_pool_buffer.template get_access<sycl::access::mode::read>(cgh);
                        auto acc_hash_values = hash_values_buffer.template get_access<sycl::access::mode::discard_write>(cgh);
                        
                        const options_type options = options_;
                        get_linear_id<data_type> get_linear_id_data{};

                        cgh.parallel_for<cut_off_points_unsorted>(sycl::range<>(attr.rank_size), [=](sycl::item<> item) {
                            const index_type idx = item.get_linear_id();

                            real_type value = 0.0;
                            for (index_type dim = 0; dim < attr.dims; ++dim) {
                                value += acc_data[get_linear_id_data(idx, dim, attr)]
                                        * acc_hash_functions[get_linear_id_hash_pool(hash_function, dim, options, attr)];
                            }
                            acc_hash_values[idx] = value;
                        });
                    });
                }

                // sort hash_values vector in a distributed fashion
                mpi::odd_even_sort(hash_values, comm_);
                
                std::vector<real_type> cut_off_points(options_.num_cut_off_points - 1, 0.0);

                // calculate cut-off points indices
                std::vector<index_type> cut_off_points_idx(cut_off_points.size());
                const index_type jump = (attr.rank_size * comm_.size()) / options_.num_cut_off_points;
                for (index_type cop = 0; cop < cut_off_points_idx.size(); ++cop) {
                    cut_off_points_idx[cop] = (cop + 1) * jump;
                }

                // fill cut-off points which are located on the current MPI rank
                for (index_type cop = 0; cop < options_.num_cut_off_points - 1; ++cop) {
                    // check if index belongs to current MPI rank
                    if (cut_off_points_idx[cop] >= attr.rank_size * comm_.rank() && cut_off_points_idx[cop] < attr.rank_size * (comm_.rank() + 1)) {
                        cut_off_points[cop] = hash_values[cut_off_points_idx[cop] % attr.rank_size];
                    }
                }

                // combine to final cut-off points on all MPI ranks
                MPI_Allreduce(MPI_IN_PLACE, cut_off_points.data(), cut_off_points.size(), mpi::type_cast<real_type>(), MPI_SUM, comm_.get());

                // copy current cut-off points to pool
                std::copy(cut_off_points.begin(), cut_off_points.end(), cut_off_points_pool.begin() + hash_function * cut_off_points.size());
            }
        }

        // select actual hash functions
        std::vector<real_type> host_buffer(options_.num_hash_tables * options_.num_hash_functions * (attr.dims + options_.num_cut_off_points - 1));
        if (comm_.master_rank()) {
            // create random generator
            #if SYCL_LSH_DEBUG
                // don't seed random engine in debug mode
                std::mt19937 rnd_uniform_gen;
            #else
                // seed random engine outside debug mode
                std::random_device rnd_device;
                std::mt19937 rnd_uniform_gen(rnd_device());
            #endif
            std::uniform_int_distribution<index_type> rnd_uniform_dist(0, options_.hash_pool_size - 1);

            get_linear_id<entropy_based<layout, options_type, data_type>> get_linear_id_functor{};

            for (index_type hash_table = 0; hash_table < options_.num_hash_tables; ++hash_table) {
                for (index_type hash_function = 0; hash_function < options_.num_hash_functions; ++hash_function) {
                    const index_type pool_hash_function = rnd_uniform_dist(rnd_uniform_gen);
                    for (index_type dim = 0; dim < attr.dims; ++dim) {
                        host_buffer[get_linear_id_functor(hash_table, hash_function, dim, options_, attr)]
                            = hash_functions_pool[get_linear_id_hash_pool(pool_hash_function, dim, options_, attr)];
                    }
                    for (index_type cop = 0; cop < options_.num_cut_off_points - 1; ++cop) {
                        host_buffer[get_linear_id_functor(hash_table, hash_function, attr.dims + cop, options_, attr)]
                            = cut_off_points_pool[pool_hash_function * (options_.num_cut_off_points - 1) + cop];
                    }
                }
            }
        }

        // broadcast hash function to other MPI ranks
        MPI_Bcast(host_buffer.data(), host_buffer.size(), mpi::type_cast<real_type>(), 0, comm_.get());

        // copy data to device buffer
        auto acc = device_buffer_.template get_access<sycl::access::mode::discard_write>();
        for (index_type i = 0; i < acc.get_count(); ++i) {
            acc[i] = host_buffer[i];
        }

        logger_.log("Created 'entropy_based' hash functions in {}.\n", t.elapsed());
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ENTROPY_BASED_HPP
