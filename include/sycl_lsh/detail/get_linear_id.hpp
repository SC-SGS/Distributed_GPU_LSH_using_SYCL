/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Defines a template class for specializing the conversion from a multidimensional index to a one-dimensional one.
 */

#ifndef SYCL_LSH_DETAIL_GET_LINEAR_ID_HPP
#define SYCL_LSH_DETAIL_GET_LINEAR_ID_HPP
#pragma once

namespace sycl_lsh::detail {

/**
 * @brief Template class to specialize the conversion from a multidimensional index to a one-dimensional.
 * @details This template class **can't** get implicitly instantiated!
 * @tparam T the type to specialize
 */
template <typename T>
struct get_linear_id;

}  // namespace sycl_lsh::detail

#endif  // SYCL_LSH_DETAIL_GET_LINEAR_ID_HPP
