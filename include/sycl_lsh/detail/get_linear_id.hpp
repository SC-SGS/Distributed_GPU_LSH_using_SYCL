/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-01
 *
 * @brief Defines a template class for specializing the conversion from a multi-dimensional index to an one-dimensional one.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_GET_LINEAR_ID_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_GET_LINEAR_ID_HPP

namespace sycl_lsh {

    /**
     * @brief Template class to specialize the conversion from a multi-dimensional index to an one-dimensional.
     * @details This template class **can't** get implicitly instantiated!
     * @tparam T the type to specialize
     */
    template <typename T>
    struct get_linear_id;

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_GET_LINEAR_ID_HPP
