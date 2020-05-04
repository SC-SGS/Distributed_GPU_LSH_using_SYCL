/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-04
 *
 * @brief Contains global constants, usings and enums.
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


#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
