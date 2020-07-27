/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-25
 *
 * @brief Contains global constants, typedefs and enums.
 */

#ifndef DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
#define DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP

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


/// Shorthand macro for std::enable_if
#define REQUIRES(cond) std::enable_if_t<(cond), int> = 0


#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
