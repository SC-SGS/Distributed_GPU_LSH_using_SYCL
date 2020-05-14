/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-14
 *
 * @brief Contains global constants, usings and enums.
 */

#ifndef DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
#define DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP

#include <type_traits>

#include <CL/sycl.hpp>


namespace sycl = cl::sycl;
/// Namespace containing helper classes and functions.
namespace detail { }


/// The memory layout.
enum class memory_layout {
    /// Array of Structs
    aos,
    /// Struct of Arrays
    soa
};


/**
 * @def REQUIRES
 * @brief A macro function for a more simple way to write a [`std::enable_if`](https://en.cppreference.com/w/cpp/types/enable_if)
 * to constraint template parameters.
 * @param[in] req a constant expression which evaluates to `true`
 */
#define REQUIRES(req) typename = std::enable_if_t<req>


#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
