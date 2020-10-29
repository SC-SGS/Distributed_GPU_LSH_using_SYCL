/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-01
 *
 * @brief Defines a template class for specializing the calculating of the hash value for a specific @ref sycl_lsh::hash_functions_type.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_LSH_HASH_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_LSH_HASH_HPP

namespace sycl_lsh {

    /**
     * @brief Template class to specialize the calculation of the hash value for a specific @ref sycl_lsh::hash_functions_type.
     * @details This template class **can't** get implicitly instantiated!
     * @tparam T the type to specialize
     */
    template <typename T>
    struct lsh_hash;

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_LSH_HASH_HPP
