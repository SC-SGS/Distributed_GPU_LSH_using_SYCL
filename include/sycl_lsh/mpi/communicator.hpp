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

#include "sycl_lsh/matrix.hpp"                // sycl_lsh::matrix, sycl_lsh::memory_layout
#include "sycl_lsh/mpi/detail/type_cast.hpp"  // sycl_lsh::mpi::detail::mpi_datatype
#include "sycl_lsh/mpi/detail/utility.hpp"    // SYCL_LSH_MPI_ERROR_CHECK

#include "mpi.h"  // MPI_Comm, MPI_COMM_WORLD, MPI_Sendrecv_replace, MPI_Gather

#include <string>  // std::string
#include <vector>  // std::vector

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

    /**
     * @brief Send-receive the @p data in a round-robin scheme, i.e., rank i sends it data to rank i + 1 and receives the data from rank i - 1.
     * @tparam T the type of the data to exchange
     * @tparam layout the matrix's memory layout
     * @param[in,out] data the sycl_lsh::matrix wrapping the data to exchange
     */
    template <typename T, memory_layout layout>
    void send_receive_round_robin(matrix<T, layout> &data) const {
        // if we only have a single MPI rank, we have nothing to do
        if (this->size() > 1) {
            const int destination = (this->rank() + 1) % this->size();
            const int source = (this->size() + (this->rank() - 1) % this->size()) % this->size();

            SYCL_LSH_MPI_ERROR_CHECK(MPI_Sendrecv_replace(data.data(), data.size(), mpi::detail::mpi_datatype<T>(), destination, 0, source, 0, comm_, MPI_STATUS_IGNORE));
        }
    }

    /**
     * @brief Gather the @p value from each MPI rank on the `communicator::main_rank()`.
     * @tparam T the type of the values to gather
     * @param[in] value the value to gather at the main MPI rank
     * @return a `std::vector` containing all gathered values (`[[nodiscard]]`)
     */
    template <typename T>
    [[nodiscard]] std::vector<T> gather(T value) const {
        // if we have only a single MPI rank, just return it without any MPI calls
        if (this->size() == 1) {
            return { value };
        }

        std::vector<T> result(this->size());
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Gather(&value, 1, detail::mpi_datatype<T>(), result.data(), 1, detail::mpi_datatype<T>(), communicator::main_rank(), comm_));
        return result;
    }

    /**
     * @brief Gather the @p str from all MPI ranks in this communicator and return it on the master rank only!
     * @param[in] str the string to retrieve on the MPI master rank from each rank
     * @return the vector of strings from each MPI rank (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<std::string> gather(const std::string &str) const;

  private:
    /// The wrapped MPI communicator.
    MPI_Comm comm_{ MPI_COMM_WORLD };
};

}  // namespace sycl_lsh::mpi

#endif  // SYCL_LSH_MPI_COMMUNICATOR_HPP
