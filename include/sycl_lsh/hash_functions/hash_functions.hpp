/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-06
 *
 * @brief Implements the factory functions for the hash functions classes.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTIONS_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTIONS_HPP

#include <sycl_lsh/memory_layout.hpp>

#include <fmt/ostream.h>

#include <ostream>

namespace sycl_lsh {

    /**
     * @brief Enum class for the different hash function types.
     */
    enum class hash_functions_type {
        /** random projections hash functions */
        random_projections,
        /** entropy based hash functions */
        entropy_based
    };

    /**
     * @brief Print the @p type of the hash functions to the output stream @p out.
     * @param[in,out] out the output stream
     * @param[in] type the hash functions type
     * @return the output stream
     */
    inline std::ostream& operator<<(std::ostream& out, const hash_functions_type type) {
        switch (type) {
            case hash_functions_type::random_projections:
                out << "random_projections";
                break;
            case hash_functions_type::entropy_based:
                out << "entropy_based";
                break;
        }
        return out;
    }


    // forward declare hash functions classes
    template <memory_layout layout, typename Options, typename Data>
    class random_projections;
    template <memory_layout layout, typename Options, typename Data>
    class entropy_based;

    namespace detail {

        /**
         * @brief Base type trait to get the actual hash functions type from the enum class @ref sycl_lsh::hash_functions_type @p type value.
         * @tparam layout the used @ref sycl_lsh::memory_layout type
         * @tparam Options the used @ref sycl_lsh::options type
         * @tparam Data the used @ref sycl_lsh::data type
         * @tparam type the @ref sycl_lsh::hash_functions_type 
         */
        template <memory_layout layout, typename Options, typename Data, hash_functions_type type>
        struct get_hash_functions_type { };

        /**
         * @brief Type trait specialization for the @ref sycl_lsh::random_projections hash functions class.
         * @tparam layout the used @ref sycl_lsh::memory_layout type
         * @tparam Options the used @ref sycl_lsh::options type
         * @tparam Data the used @ref sycl_lsh::data type
         */
        template <memory_layout layout, typename Options, typename Data>
        struct get_hash_functions_type<layout, Options, Data, hash_functions_type::random_projections> {
            using type = random_projections<layout, Options, Data>;
        };

        /**
         * @brief Type trait specialization for the @ref sycl_lsh::entropy_based hash functions class.
         * @tparam layout the used @ref sycl_lsh::memory_layout type
         * @tparam Options the used @ref sycl_lsh::options type
         * @tparam Data the used @ref sycl_lsh::data type
         */
        template <memory_layout layout, typename Options, typename Data>
        struct get_hash_functions_type<layout, Options, Data, hash_functions_type::entropy_based> {
            using type = entropy_based<layout, Options, Data>;
        };

        /**
         * @brief Type alias for the @ref sycl_lsh::detail::get_hash_functions_type type trait class.
         */
        template <memory_layout layout, typename Options, typename Data, hash_functions_type type>
        using get_hash_functions_type_t = typename get_hash_functions_type<layout, Options, Data, type>::type;

    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTIONS_HPP
