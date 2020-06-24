/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-24
 * @brief Timing macros compatible with MPI and/or SYCL.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_TIMING_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_TIMING_HPP


#include <chrono>

#include <config.hpp>
#include <detail/print.hpp>


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
 * @def END_TIMING_MPI
 * @brief A macro function to end timing and print the elapsed time on a specific MPI rank.
 * @param[in] name the name of the currently timed functionality
 * @param[in] comm_rank the current MPI rank
 *
 * @attention Before calling `END_TIMING_MPI(x, comm_rank)` a call to `START_TIMING(x)` **must** be made!
 *
 *
 * @def END_TIMING_BARRIER
 * @brief A macro function to end timing and print the elapsed time. Calls `sycl::queue::wait()` before timing.
 * @param[in] name the name of the currently timed functionality
 * @param[in] queue the SYCL queue to call `wait()` on
 *
 * @attention Before calling `END_TIMING_BARRIER(x, queue)` a call to `START_TIMING(x)` **must** be made!
 *
 *
 * @def END_TIMING_MPI_AND_BARRIER
 * @brief A macro function to end timing and print the elapsed time on a specific MPI rank. Calls `sycl::queue::wait()` before timing.
 * @param[in] name the name of the currently timed functionality
 * @param[in] comm_rank the current MPI rank
 * @param[in] queue the SYCL queue to call `wait()` on
 *
 * @attention Before calling `END_TIMING_MPI_AND_BARRIER(x, comm_rank, queue)` a call to `START_TIMING(x)` **must** be made!
 */
#ifdef ENABLE_TIMING
#define START_TIMING(name) const auto start_##name = std::chrono::steady_clock::now();

#define END_TIMING(name)                                                                                               \
do {                                                                                                                   \
const auto end_##name = std::chrono::steady_clock::now();                                                              \
const auto duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count(); \
detail::print("Elapsed time ({}): {} ms\n", #name, duration_##name);                                                   \
} while (false)

#define END_TIMING_MPI(name, comm_rank)                                                                                \
do {                                                                                                                   \
const auto end_##name = std::chrono::steady_clock::now();                                                              \
const auto duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count(); \
detail::mpi_print<print_rank>(comm_rank, "Elapsed time on rank {} ({}): {} ms\n", comm_rank, #name, duration_##name);  \
} while (false)

#define END_TIMING_BARRIER(name, queue)                                                                                \
do {                                                                                                                   \
queue.wait_and_throw();                                                                                                          \
const auto end_##name = std::chrono::steady_clock::now();                                                              \
const auto duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count(); \
detail::print("Elapsed time ({}): {} ms\n", #name, duration_##name);                                                   \
} while (false)

#define END_TIMING_MPI_AND_BARRIER(name, comm_rank, queue)                                                             \
do {                                                                                                                   \
queue.wait_and_throw();                                                                                                          \
const auto end_##name = std::chrono::steady_clock::now();                                                              \
const auto duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count(); \
detail::mpi_print<print_rank>(comm_rank, "Elapsed time on rank {} ({}): {} ms\n", comm_rank, #name, duration_##name);  \
} while (false)
#else
#define START_TIMING(name)
#define END_TIMING(name)
#define END_TIMING_MPI(name, comm_rank)
#define END_TIMING_BARRIER(name, queue)
#define END_TIMING_MPI_AND_BARRIER(name, comm_rank, queue)
#endif

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_TIMING_HPP
