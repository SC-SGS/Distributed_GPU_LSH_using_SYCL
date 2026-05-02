/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the factory functions for the hash functions classes.
 */

#ifndef SYCL_LSH_HASH_FUNCTIONS_HASH_FUNCTIONS_HPP
#define SYCL_LSH_HASH_FUNCTIONS_HASH_FUNCTIONS_HPP
#pragma once

#include "sycl_lsh/hash_functions/hash_function_types.hpp"  // sycl_lsh::hash_function_type
#include "sycl_lsh/memory_layout.hpp"                       // sycl_lsh::memory_layout

namespace sycl_lsh {

// forward declare hash functions classes
template <memory_layout layout, typename Data>
class random_projections;
template <memory_layout layout, typename Data>
class entropy_based;
template <memory_layout layout, typename Data>
class mixed_hash_functions;

namespace detail {

/**
 * @brief Base type trait to get the actual hash function type from the enum class @ref sycl_lsh::hash_function_type @p type value.
 * @tparam layout the used @ref sycl_lsh::memory_layout type
 * @tparam Data the used @ref sycl_lsh::data type
 * @tparam type the @ref sycl_lsh::hash_function_type
 */
template <memory_layout layout, typename Data, hash_function_type type>
struct get_hash_function_type {};

/**
 * @brief Type trait specialization for the @ref sycl_lsh::random_projections hash functions class.
 * @tparam layout the used @ref sycl_lsh::memory_layout type
 * @tparam Data the used @ref sycl_lsh::data type
 */
template <memory_layout layout, typename Data>
struct get_hash_function_type<layout, Data, hash_function_type::random_projections> {
    using type = random_projections<layout, Data>;
};

/**
 * @brief Type trait specialization for the @ref sycl_lsh::entropy_based hash functions class.
 * @tparam layout the used @ref sycl_lsh::memory_layout type
 * @tparam Data the used @ref sycl_lsh::data type
 */
template <memory_layout layout, typename Data>
struct get_hash_function_type<layout, Data, hash_function_type::entropy_based> {
    using type = entropy_based<layout, Data>;
};

/**
 * @brief Type trait specialization for the @ref sycl_lsh::mixed_hash_functions hash functions class.
 * @tparam layout the used @ref sycl_lsh::memory_layout type
 * @tparam Data the used @ref sycl_lsh::data type
 */
template <memory_layout layout, typename Data>
struct get_hash_function_type<layout, Data, hash_function_type::mixed_hash_functions> {
    using type = mixed_hash_functions<layout, Data>;
};

/**
 * @brief Type alias for the @ref sycl_lsh::detail::get_hash_function_type type trait class.
 */
template <memory_layout layout, typename Data, hash_function_type type>
using get_hash_function_type_t = typename get_hash_function_type<layout, Data, type>::type;

}  // namespace detail

}  // namespace sycl_lsh

#endif  // SYCL_LSH_HASH_FUNCTIONS_HASH_FUNCTIONS_HPP
