/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Wrapper function to automatically initialize and finalize the MPI environment correctly.
 */

#ifndef SYCL_LSH_MPI_MAIN_HPP
#define SYCL_LSH_MPI_MAIN_HPP
#pragma once

namespace sycl_lsh::mpi {

/// The type of the custom main function called inside @ref sycl_lsh::mpi::main().
using custom_main_ptr = int (*)(int, char **);

/**
 * @brief Initializes and finalizes the MPI environment with the required level of thread support (*MPI_THREAD_SERIALIZED*)
 *        and calls the custom main function denoted by @p func.
 * @param[in] argc the number of command line arguments
 * @param[in] argv the command line arguments
 * @param[in] func the custom main function to call
 * @return the return code of @p func or [*EXIT_FAILURE*](https://en.cppreference.com/w/cpp/utility/program/EXIT_status) if the
 *         required level of thread support couldn't be satisfied
 */
int main(int argc, char **argv, custom_main_ptr func);

}  // namespace sycl_lsh::mpi

#endif  // SYCL_LSH_MPI_MAIN_HPP
