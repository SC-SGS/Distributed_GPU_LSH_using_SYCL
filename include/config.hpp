/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-17
 *
 * @brief Contains global constants, typedefs and enums.
 */

#ifndef DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
#define DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP

#include <chrono>
#include <type_traits>

#include <CL/sycl.hpp>

#include <detail/print.hpp>


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


/**
 * @def START_TIMING
 * @brief A macro function to start timing.
 * @param[in] name the name of the currently timed functionality
 *
 *
 * @def END_TIMING
 * @brief A macro function to end timing and print the elapsed time.
 * @param[in] name the name of the currently timed functionality
 *
 * @attention Before calling `END_TIMING(x)` a call to `START_TIMING(x)` **must** be made!
 *
 *
 * @def END_TIMING_WITH_MPI
 * @brief A macro function to end timing and print the elapsed time on a specific MPI rank.
 * @param[in] name the name of the currently timed functionality
 * @param[in] communicator the *MPI_Comm* communicator
 *
 * @attention Before calling `END_TIMING_WITH_MPI(x, comm)` a call to `START_TIMING(x)` **must** be made!
 *
 *
 * @def END_TIMING_WITH_BARRIER
 * @brief A macro function to end timing and print the elapsed time. Calls `sycl::queue::wait()` before timing.
 * @param[in] name the name of the currently timed functionality
 * @param[in] queue the SYCL queue to call `wait()` on
 *
 * @attention Before calling `END_TIMING_WITH_BARRIER(x, queue)` a call to `START_TIMING(x)` **must** be made!
 *
 *
 * @def END_TIMING_WITH_MPI_AND_BARRIER
 * @brief A macro function to end timing and print the elapsed time on a specific MPI rank. Calls `sycl::queue::wait()` before timing.
 * @param[in] name the name of the currently timed functionality
 * @param[in] communicator the *MPI_Comm* communicator
 * @param[in] queue the SYCL queue to call `wait()` on
 *
 * @attention Before calling `END_TIMING_WITH_MPI_AND_BARRIER(x, comm, queue)` a call to `START_TIMING(x)` **must** be made!
 */
#ifdef ENABLE_TIMING
#define START_TIMING(name) const auto start_##name = std::chrono::steady_clock::now();

#define END_TIMING(name)                                                                                                \
do {                                                                                                                    \
const auto end_##name = std::chrono::steady_clock::now();                                                               \
const auto duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count();  \
detail::print("Elapsed time ({}): {} ms\n", #name, duration_##name);                                                    \
} while (false)

#define END_TIMING_WITH_MPI(name, communicator)                                                                                     \
do {                                                                                                                                \
const auto end_##name = std::chrono::steady_clock::now();                                                                           \
const auto duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count();              \
int comm_rank_##name;                                                                                                               \
MPI_Comm_rank(communicator, &comm_rank_##name);                                                                                     \
detail::mpi_print<print_rank>(comm_rank_##name, "Elapsed time on rank {} ({}): {} ms\n", comm_rank_##name, #name, duration_##name); \
} while (false)

#define END_TIMING_WITH_BARRIER(name, queue)                                                                            \
do {                                                                                                                    \
queue.wait();                                                                                                           \
const auto end_##name = std::chrono::steady_clock::now();                                                               \
const auto duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count();  \
detail::print("Elapsed time ({}): {} ms\n", #name, duration_##name);                                                    \
} while (false)

#define END_TIMING_WITH_MPI_AND_BARRIER(name, communicator, queue)                                                                  \
do {                                                                                                                                \
queue.wait();                                                                                                                       \
const auto end_##name = std::chrono::steady_clock::now();                                                                           \
const auto duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count();              \
int comm_rank_##name;                                                                                                               \
MPI_Comm_rank(communicator, &comm_rank_##name);                                                                                     \
detail::mpi_print<print_rank>(comm_rank_##name, "Elapsed time on rank {} ({}): {} ms\n", comm_rank_##name, #name, duration_##name); \
} while (false)
#else
#define START_TIMING(name)
#define END_TIMING(name)
#define END_TIMING_WITH_MPI(name, communicator)
#define END_TIMING_WITH_BARRIER(name, queue)
#define END_TIMING_WITH_MPI_AND_BARRIER(name, communicator, queue)
#endif

#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
