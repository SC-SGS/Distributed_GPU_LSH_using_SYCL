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

#include "mpi.h"  // MPI_Comm related functionality

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
     * @brief Construct a new @ref sycl_lsh::mpi::communicator as copy of *MPI_COMM_WORLD*.
     */
    communicator();
    /**
     * @brief Construct a new @ref sycl_lsh::mpi::communicator as a copy of @p other.
     * @param[in] other the @ref sycl_lsh::mpi::communicator to copy
     */
    communicator(const communicator &other);
    /**
     * @brief Construct a new @ref sycl_lsh::mpi::communicator from the resources hold by @p other.
     * @param[in,out] other the @ref sycl_lsh::mpi::communicator to move-from
     */
    communicator(communicator &&other) noexcept;
    /**
     * @brief Construct a new @ref sycl_lsh::mpi::communicator from the given MPI_Comm.
     * @param[in] comm the MPI_Comm to wrap
     * @param[in] is_freeable `true` if @p comm should be freed at the end of `*this` lifetime, `false` otherwise
     */
    communicator(MPI_Comm comm, bool is_freeable) noexcept;
    /**
     * @brief Destruct the @ref sycl_lsh::mpi::communicator object.
     * @details Only calls *MPI_Comm_free* if @ref sycl_lsh::mpi::communicator::is_freeable() returns `true`.
     */
    ~communicator();

    // ---------------------------------------------------------------------------------------------------------- //
    //                                            assignment operators                                            //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Copy-assigns @p rhs to `*this`.
     * @param[in] rhs the @ref sycl_lsh::mpi::communicator to copy
     * @return `*this`
     */
    communicator &operator=(const communicator &rhs);
    /**
     * @brief Move-assigns @p rhs to `*this`.
     * @param[in] rhs the @ref sycl_lsh::mpi::communicator to move-from
     * @return `*this`
     */
    communicator &operator=(communicator &&rhs) noexcept;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                         MPI communicator functions                                         //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Returns the current MPI rank.
     * @return the rank (`[[nodiscard]]`)
     */
    [[nodiscard]] int rank() const;
    /**
     * @brief Returns the size of the MPI communicator.
     * @return the communicator size (`[[nodiscard]]`)
     */
    [[nodiscard]] int size() const;
    /**
     * @brief Return the MPI main rank, i.e., MPI rank 0 in the current communicator.
     * @return the main rank: 0 (`[[nodiscard]]`)
     */
    [[nodiscard]] static constexpr int main_rank() noexcept { return 0; }
    /**
     * @brief Returns `true` if the current MPI rank is the main rank, i.e., MPI rank 0.
     * @return `true` if the current MPI rank is rank 0, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_main_rank() const;
    /**
     * @brief Waits for all MPI processes in this communicator.
     */
    void barrier() const;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                   getter                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Get the underlying MPI communicator.
     * @return the MPI communicator wrapped in this @ref sycl_lsh::mpi::communicator object (`[[nodiscard]]`)
     */
    [[nodiscard]] const MPI_Comm &get() const noexcept { return comm_; }
    /**
     * @brief Get the underlying MPI communicator.
     * @return the MPI communicator wrapped in this @ref sycl_lsh::mpi::communicator object (`[[nodiscard]]`)
     */
    [[nodiscard]] MPI_Comm &get() noexcept { return comm_; }
    /**
     * @brief Returns whether the underlying MPI communicator gets automatically freed upon destruction.
     * @return `true` if *MPI_Comm_free* gets called upon destruction, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_freeable() const noexcept { return is_freeable_; }

  private:
    /// The wrapped MPI communicator.
    MPI_Comm comm_;
    /// A flag deciding whether the MPI_Comm should be freed upon destruction of *this or not.
    bool is_freeable_;
};

}  // namespace sycl_lsh::mpi

#endif  // SYCL_LSH_MPI_COMMUNICATOR_HPP
