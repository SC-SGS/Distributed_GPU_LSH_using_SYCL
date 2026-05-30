/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Minimalistic wrapper class around an MPI communicator.
 */

#ifndef SYCL_LSH_MPI_COMMUNICATOR_HPP
#define SYCL_LSH_MPI_COMMUNICATOR_HPP
#pragma once

#include "mpi.h"  // MPI_Comm, MPI_COMM_WORLD

namespace sycl_lsh::mpi {

/**
 * @brief Minimalistic wrapper around an MPI communicator.
 */
class communicator {
  public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                        constructors and destructor                                         //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::mpi::communicator wrapping MPI_COMM_WORLD.
     */
    communicator() = default;
    /**
     * @brief Construct a new @ref sycl_lsh::mpi::communicator from the given MPI_Comm.
     * @param[in] comm the MPI_Comm to wrap
     * @note This function does not take ownership of the provided MPI communicator!
     */
    explicit communicator(MPI_Comm comm) noexcept;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                         MPI communicator functions                                         //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Returns the size of the MPI communicator.
     * @return the communicator size (`[[nodiscard]]`)
     */
    [[nodiscard]] int size() const;
    /**
     * @brief Returns the current MPI rank.
     * @return the rank (`[[nodiscard]]`)
     */
    [[nodiscard]] int rank() const;

    /**
     * @brief Return the MPI main rank, i.e., MPI rank 0 in the current communicator.
     * @return the main rank: 0 (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr static int main_rank() noexcept { return 0; }

    /**
     * @brief Returns `true` if the current MPI rank is the main rank, i.e., MPI rank 0.
     * @return `true` if the current MPI rank is rank 0, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_main_rank() const;
    /**
     * @brief Waits for all MPI processes in this communicator.
     */
    void barrier() const;

    /**
     * @brief Add implicit conversion operator back to a native MPI communicator.
     * @return The wrapped MPI communicator (`[[nodiscard]]`)
     */
    [[nodiscard]] operator MPI_Comm() const { return comm_; }  // NOLINT: implicit conversion desired

  private:
    /// The wrapped MPI communicator.
    MPI_Comm comm_{ MPI_COMM_WORLD };
};

}  // namespace sycl_lsh::mpi

#endif  // SYCL_LSH_MPI_COMMUNICATOR_HPP
