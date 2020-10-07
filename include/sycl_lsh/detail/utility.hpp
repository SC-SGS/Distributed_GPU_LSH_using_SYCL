/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-07
 *
 * @brief Different utility functions or macros.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP

/**
 * @brief Shorthand macro for an easier [`std::enable_if`](https://en.cppreference.com/w/cpp/types/enable_if).
 */
#define SYCL_LSH_REQUIRES(cond) std::enable_if_t<(cond), int> = 0

namespace sycl_lsh::detail {

    /**
     * @brief Swaps the values of @p lhs and @p rhs.
     * @tparam T the type of the elements to swap
     * @param[in,out] lhs the first value to swap
     * @param[in,out] rhs the second value to swap
     */
    template <typename T>
    inline void swap(T& lhs, T& rhs) {
        const T tmp = lhs;
        lhs = rhs;
        rhs = tmp;
    }


}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP
