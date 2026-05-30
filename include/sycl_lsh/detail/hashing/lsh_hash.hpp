/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Defines a template class for specializing the calculating of the hash value for a specific @ref sycl_lsh::hash_functions_type.
 */

#ifndef SYCL_LSH_DETAIL_HASHING_LSH_HASH_HPP
#define SYCL_LSH_DETAIL_HASHING_LSH_HASH_HPP
#pragma once

namespace sycl_lsh::detail::hashing {

/**
 * @brief Template class to specialize the calculation of the hash value for a specific @ref sycl_lsh::hash_functions_type.
 * @details This template class **can't** get implicitly instantiated!
 * @tparam T the type to specialize
 */
template <typename T>
struct lsh_hash;

}  // namespace sycl_lsh::detail::hashing

#endif  // SYCL_LSH_DETAIL_HASHING_LSH_HASH_HPP
