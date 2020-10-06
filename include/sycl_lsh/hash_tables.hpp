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

namespace sycl_lsh {

    // forward declare hash_tables class
    template <memory_layout layout, typename Options, typename Data, template <memory_layout, typename, typename> class HashFunctionType>
    class hash_tables;

    template <memory_layout layout, typename Options, typename Data>
    auto make_hash_tables(const Options& opt, Data& data, const mpi::communicator& comm, const mpi::logger& logger) {
        return hash_tables<layout, Options, Data, random_projections>(opt, data, comm, logger);
    }


    template <memory_layout layout, typename Options, typename Data, template <memory_layout, typename, typename> class HashFunctionType>
    class hash_tables : private detail::hash_tables_base {
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
        using hash_function_type = HashFunctionType<layout, options_type, data_type>;

        /// The type of the device buffer used by SYCL.
        using device_buffer_type = sycl::buffer<index_type, 1>;


    private:
        // befriend factory function
        friend auto make_hash_tables<layout, Options, Data>(const options_type&, data_type&, const mpi::communicator&, const mpi::logger&);

        // ---------------------------------------------------------------------------------------------------------- //
        //                                                constructor                                                 //
        // ---------------------------------------------------------------------------------------------------------- //
        hash_tables(const options_type& opt, data_type& data, const mpi::communicator& comm, const mpi::logger& logger)
            : options_(opt), data_(data), attr_(data.get_attributes()), comm_(comm), logger_(logger),
              hash_functions_(opt, data, comm, logger),
              queue_(sycl::default_selector()), // TODO 2020-10-06 14:08 marcel: change to custom selector
              hash_tables_buffer_(opt.num_hash_tables * data.get_attributes().rank_size + options_type::blocking_size), // TODO 2020-10-06 14:14 marcel: check blocking
              offsets_buffer_(opt.num_hash_tables * (opt.hash_table_size + 1))
        {

        }


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

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLES_HPP
