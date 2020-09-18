/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-18
 *
 * @brief Different utility functions or macros.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP

/**
 * @brief Shorthand macro for an easier [`std::enable_if`](https://en.cppreference.com/w/cpp/types/enable_if).
 */
#define SYCL_LSH_REQUIRES(cond) std::enable_if_t<(cond), int> = 0

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP
