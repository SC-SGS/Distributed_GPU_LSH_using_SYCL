/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-06
 *
 * @brief Implements the @ref hash_tables class representing the used LSH hash tables.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLES_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLES_HPP

#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/hash_functions/hash_functions.hpp>
#include <sycl_lsh/memory_layout.hpp>

#include <type_traits>
#include <vector>

namespace sycl_lsh {

    // SYCL kernel name needed to silence ComputeCpp warning
    class kernel_count_hash_values;
    class kernel_calculate_offsets;
    class kernel_fill_hash_tables;

    // forward declare hash_tables class
    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    class hash_tables;

    /**
     * @brief Factory function for the @ref sycl_lsh::hash_tables class.
     * @brief Used to be able to automatically deduce the @ref sycl_lsh::options and @ref sycl_lsh::data types.
     * @tparam layout the used @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     * @tparam Data the used @ref sycl_lsh::data type
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] data the used @ref sycl_lsh::data representing the used data set
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     * @return the @ref sycl_lsh::hash_tables object representing the hash tables used in the LSH algorithm (`[[nodiscard]]`)
     */
    template <memory_layout layout, typename Options, typename Data>
    [[nodiscard]] 
    auto make_hash_tables(const Options& opt, Data& data, const mpi::communicator& comm, const mpi::logger& logger) {
        using type_of_hash_functions = detail::get_hash_functions_type_t<layout, Options, Data, Options::type_of_hash_functions>;
        return hash_tables<layout, Options, Data, type_of_hash_functions>(opt, data, comm, logger);
    }


    /**
     * @brief Class which represents the hash tables used om the LSH algorithm. Performs the actual calculation of the k-nearest-neighbors.
     * @tparam layout the @ref sycl_lsh::memory_layout type
     * @tparam Options the used @ref sycl_lsh::options type
     * @tparam Data the used @ref sycl_lsh::data type
     * @tparam HashFunctionType the used type of hash functions in the LSH algorithm
     */
    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    class hash_tables : private detail::hash_tables_base {
        // ---------------------------------------------------------------------------------------------------------- //
        //                                      template parameter sanity checks                                      //
        // ---------------------------------------------------------------------------------------------------------- //
        static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must be a sycl_lsh::options type!");
        static_assert(std::is_base_of_v<detail::data_base, Data>, "The third template parameter must be a sycl_lsh::data type!");
        static_assert(std::is_base_of_v<detail::hash_functions_base, HashFunctionType>, "The fourth template parameter must be a hash function type!");
    public:
        // ---------------------------------------------------------------------------------------------------------- //
        //                                                type aliases                                                //
        // ---------------------------------------------------------------------------------------------------------- //
        /// The type of the @ref sycl_lsh::options object.
        using options_type = Options;
        /// The used floating point type.
        using real_type = typename options_type::real_type;
        /// The used integral type (used for indices).
        using index_type = typename options_type::index_type;
        /// The used unsigned type (used for the hash values).
        using hash_value_type = typename options_type::hash_value_type;

        /// The type of the @ref sycl_lsh::data object.
        using data_type = Data;
        /// The type of the @ref sycl_lsh::data_attributes object.
        using data_attributes_type = typename data_type::data_attributes_type;

        /// The type of the used LSH hash functions.
        using hash_function_type = HashFunctionType;

        /// The type of the device buffer used by SYCL.
        using device_buffer_type = sycl::buffer<index_type, 1>;


        // ---------------------------------------------------------------------------------------------------------- //
        //                                       calculate k-nearest-neighbors                                        //
        // ---------------------------------------------------------------------------------------------------------- //



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
        const data_type& get_data() const noexcept { return data_; }

    private:
        // befriend factory function
        friend auto make_hash_tables<layout, Options, Data>(const options_type&, data_type&, const mpi::communicator&, const mpi::logger&);

        // ---------------------------------------------------------------------------------------------------------- //
        //                                                constructor                                                 //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Constructs a new @ref sycl_lsh::hash_tables object initializing the LSH hash tables.
         * @param[in] opt the used @ref sycl_lsh::options
         * @param[in] data the used @ref sycl_lsh::data representing the used data set
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         * @param[in] logger the used @ref sycl_lsh::mpi::logger
         */
        hash_tables(const options_type& opt, data_type& data, const mpi::communicator& comm, const mpi::logger& logger);

        /**
         * @brief Calculate the number of data points assigned to each hash bucket in each hash table.
         * @param[in,out] hash_values_count the number of data points per hash bucket
         */
        void count_hash_values(device_buffer_type& hash_values_count);
        /**
         * @brief Calculates the offset of each hash bucket in each hash table.
         * @param[in] hash_values_count the number of data points per hash bucket
         */
        void calculate_offsets(device_buffer_type& hash_values_count);
        /**
         * @brief Fill each hash table based on the previously calculated offsets.
         */
        void fill_hash_tables();


        const options_type& options_;
        data_type& data_;
        const data_attributes_type attr_;
        const mpi::communicator& comm_;
        const mpi::logger& logger_;

        hash_function_type hash_functions_;

        sycl::queue queue_;
        device_buffer_type hash_tables_buffer_;
        device_buffer_type offsets_buffer_;

    };


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    hash_tables<layout, Options, Data, HashFunctionType>::hash_tables(const Options& opt, Data& data, const mpi::communicator& comm, const mpi::logger& logger)
            : options_(opt), data_(data), attr_(data.get_attributes()), comm_(comm), logger_(logger),
              hash_functions_(opt, data, comm, logger),
              queue_(sycl::default_selector()), // TODO 2020-10-06 14:08 marcel: change to custom selector
              hash_tables_buffer_(opt.num_hash_tables * data.get_attributes().rank_size + options_type::blocking_size), // TODO 2020-10-06 14:14 marcel: check blocking
              offsets_buffer_(opt.num_hash_tables * (opt.hash_table_size + 1))
    {
        // TODO 2020-10-06 16:25 marcel: use correct timer overload
        mpi::timer t(comm_);

        {
            // create temporary buffer to count the occurrence of each hash value
            device_buffer_type hash_values_count(options_.num_hash_tables * options_.hash_table_size);

            // count the occurrence of each hash value per hash table
            this->count_hash_values(hash_values_count);
            // calculate the offset values
            this->calculate_offsets(hash_values_count);
        }
        // fill the hash tables based on the previously calculated offsets
        this->fill_hash_tables();

        logger_.log("Created hash tables in {}.\n", t.elapsed());
    }

    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    void hash_tables<layout, Options, Data, HashFunctionType>::count_hash_values(device_buffer_type& hash_values_count) {
        // TODO 2020-10-06 16:25 marcel: use correct timer overload
        mpi::timer t(comm_);

        queue_.submit([&](sycl::handler& cgh) {
            // get accessors
            auto acc_hash_values_count = hash_values_count.template get_access<sycl::access::mode::atomic>(cgh);
            auto acc_hash_functions = hash_functions_.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
            auto acc_data = data_.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
            // get additional information
            auto options = options_;
            auto attr = attr_;
            // get hasher functor instantiation
            const lsh_hash<hash_function_type> hasher{};

            cgh.parallel_for<kernel_count_hash_values>(sycl::range<>(attr.rank_size), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                for (index_type hash_table = 0; hash_table < options.num_hash_tables; ++hash_table) {
                    const hash_value_type hash_value = hasher(hash_table, idx, acc_data, acc_hash_functions, options, attr);
                    acc_hash_values_count[hash_table * options.hash_table_size + hash_value].fetch_add(1);
                }
            });
        });

        logger_.log("Counted hash values in {}.\n", t.elapsed());
    }
    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    void hash_tables<layout, Options, Data, HashFunctionType>::calculate_offsets(device_buffer_type& hash_values_count) {
        // TODO 2020-10-06 16:25 marcel: use correct timer overload
        mpi::timer t(comm_);

        queue_.submit([&](sycl::handler& cgh) {
            // get accessors
            auto acc_hash_values_count = hash_values_count.template get_access<sycl::access::mode::read>(cgh);
            auto acc_offset = offsets_buffer_.template get_access<sycl::access::mode::discard_write>(cgh);
            // get additional information
            auto options = options_;

            cgh.parallel_for<kernel_calculate_offsets>(sycl::range<>(options.num_hash_tables), [=](sycl::item<> item){
                const index_type idx = item.get_linear_id();

                // calculate constant offsets
                const index_type hash_table_offset = idx * (options.hash_table_size + 1);
                const index_type hash_value_count_offset = idx * options.hash_table_size;
                // zero out the first two offsets in each hash table
                acc_offset[hash_table_offset] = 0;
                acc_offset[hash_table_offset + 1] = 0;
                // fill remaining offset values
                for (index_type hash_value = 2; hash_value <= options.hash_table_size; ++hash_value) {
                    // calculated modified prefix sum
                    acc_offset[hash_table_offset + hash_value] =
                            acc_offset[hash_table_offset + hash_value - 1] +
                            acc_hash_values_count[hash_value_count_offset + hash_value - 2];
                }
            });
        });

        logger_.log("Calculated offsets in {}.\n", t.elapsed());
    }
    template <memory_layout layout, typename Options, typename Data, typename HashFunctionType>
    void hash_tables<layout, Options, Data, HashFunctionType>::fill_hash_tables() {
        // TODO 2020-10-06 16:25 marcel: use correct timer overload
        mpi::timer t(comm_);

        queue_.submit([&](sycl::handler& cgh){
            // get accessors
            auto acc_data = data_.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
            auto acc_hash_functions = hash_functions_.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
            auto acc_offsets = offsets_buffer_.template get_access<sycl::access::mode::atomic>(cgh);
            auto acc_hash_tables = hash_tables_buffer_.template get_access<sycl::access::mode::discard_write>(cgh);
            // get additional information
            auto options = options_;
            auto attr = attr_;
            const index_type base_id = comm_.rank() * attr.rank_size;
            const index_type comm_rank = comm_.rank();
            const index_type comm_size = comm_.size();
            // get hasher functor instantiation
            const lsh_hash<hash_function_type> hasher{};

            cgh.parallel_for<kernel_fill_hash_tables>(sycl::range<>(attr.rank_size), [=](sycl::item<> item){
                const index_type idx = item.get_linear_id();

                // TODO 2020-10-06 16:56 marcel: better?
                index_type val = base_id + idx;
                if (comm_rank == comm_size - 1) {
                    const index_type correct_rank_size = attr.total_size - ((comm_size - 1) * attr.rank_size);
                    if (idx >= correct_rank_size) {
                        val = base_id + correct_rank_size - 1;
                    }
                }

                for (index_type hash_table = 0; hash_table < options.num_hash_tables; ++hash_table) {
                    // get hash value
                    const hash_value_type hash_value = hasher(hash_table, idx, acc_data, acc_hash_functions, options, attr);
                    // update offsets
                    const index_type hash_table_idx = acc_offsets[hash_table * (options.hash_table_size + 1) + hash_value + 1].fetch_add(1);
                    acc_hash_tables[hash_table * attr.rank_size + hash_table_idx] = val;
                }

                // TODO 2020-10-06 16:59 marcel: better?
                // fill additional values needed for blocking
                if (idx == attr.rank_size - 1) {
                    for (index_type block = 0; block < options_type::blocking_size; ++block) {
                        acc_hash_tables[options.num_hash_tables * attr.rank_size + block] = val;
                    }
                }
            });
        });

        logger_.log("Filled hash tables in {}.\n", t.elapsed());
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLES_HPP
