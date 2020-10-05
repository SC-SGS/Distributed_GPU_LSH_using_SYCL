/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-05
 *
 * @brief Implements the @ref hash_tables class representing the used LSH hash tables.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLES_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLES_HPP

#include <sycl_lsh/detail/defines.hpp>
#include <sycl_lsh/detail/sycl.hpp>
#include <sycl_lsh/memory_layout.hpp>

namespace sycl_lsh {

    template <memory_layout layout, typename Options, typename Data, template <memory_layout, typename, typename> HashFunctionType>
    class hash_tables : detail::hash_tables_base {
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
        using options_type = Options
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


    private:

        const options_type& options_;
        data_type& data_;
        const data_attributes_type attr_;
        const mpi::communicator& comm_;
        const mpi::logger& logger_;

        hash_function_type hash_functions_;

        device_buffer_type hash_tables_buffer_;
        device_buffer_type offsets_buffer_;

    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLES_HPP
