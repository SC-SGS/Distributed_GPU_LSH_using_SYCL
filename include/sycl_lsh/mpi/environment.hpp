/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Defines some wrapper functions around MPI specific environment functions.
 */

#ifndef SYCL_LSH_MPI_ENVIRONMENT_HPP
#define SYCL_LSH_MPI_ENVIRONMENT_HPP
#pragma once

namespace sycl_lsh::mpi {

/**
 * @brief Check if the MPI environment has been successfully initialized.
 * @return `true` if the environment was successfully initialized, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool is_initialized();

/**
 * @brief Check if the MPI environment has been successfully finalized.
 * @return `true` if the environment was successfully finalized, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool is_finalized();

/**
 * @brief Check if the MPI environment is currently active, i.e., `init` has already been called, but not `finalize`.
 * @return `true` if the environment is currently active, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool is_active();

}  // namespace sycl_lsh::mpi

#endif  // SYCL_LSH_MPI_ENVIRONMENT_HPP
