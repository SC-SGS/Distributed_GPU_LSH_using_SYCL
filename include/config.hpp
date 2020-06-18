/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-18
 *
 * @brief Contains global constants, typedefs and enums.
 */

#ifndef DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
#define DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP

#include <chrono>
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


/// The MPI rank on which the @ref detail::mpi_print information should be printed.
constexpr int print_rank = 0;



#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
