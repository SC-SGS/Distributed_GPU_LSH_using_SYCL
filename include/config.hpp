/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-26
 *
 * @brief Contains global constants, usings and enums.
 */

#ifndef DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
#define DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP

#include <chrono>
#include <iostream>
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
 * @def START_TIMING
 * @brief A macro function to start timing.
 * @param[in] name the name of the currently timed functionality
 *
 * @def END_TIMING
 * @brief A macro function to end timing and print the elapsed time.
 * @param[in] name the name of the currently timed functionality
 *
 * @attention Before calling `END_TIMING(x)` a call to `START_TIMING(x)` **must** be made!
 *
 * @def END_TIMING_WITH_BARRIER
 * @brief A macro function to end timing and print the elapsed time. Calls `sycl::queue::wait()` before timing.
 * @param[in] name the name of the currently timed functionality
 * @param[in] queue the SYCL queue to call `wait()` on
 *
 * @attention Before calling `END_TIMING(x)` or `END_TIMING_WITH_BARRIER(x, queue)` a call to `START_TIMING(x)` **must** be made!
 */
#ifdef ENABLE_TIMING
#define START_TIMING(name) const auto start_##name = std::chrono::steady_clock::now();

#define END_TIMING(name)                                                                                                \
do {                                                                                                                    \
const auto end_##name = std::chrono::steady_clock::now();                                                               \
const auto duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count();  \
std::cout << "Elapsed time (" << #name << "): " << duration_##name << " ms" << std::endl;                               \
} while (false)

#define END_TIMING_WITH_BARRIER(name, queue)                                                                            \
do {                                                                                                                    \
queue.wait();                                                                                                           \
const auto end_##name = std::chrono::steady_clock::now();                                                               \
const auto duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count();  \
std::cout << "Elapsed time (" << #name << "): " << duration_##name << " ms" << std::endl;                               \
} while (false)
#else
#define START_TIMING(name)
#define END_TIMING(name)
#define END_TIMING_WITH_BARRIER(name, queue)
#endif


#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
