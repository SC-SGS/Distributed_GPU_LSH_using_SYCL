/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-22
 *
 * @brief Exception class for errors occurring in @ref sycl_lsh::mpi::communicator.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_COMMUNICATOR_EXCEPTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_COMMUNICATOR_EXCEPTION_HPP

#include <exception>

#include <mpi.h>

namespace sycl_lsh::mpi {

    /**
     * @brief Exception class for errors occurring in @ref sycl_lsh::mpi::communicator.
     */
    class communicator_exception : public std::exception {
    public:
        /**
         * @brief Constructs a new @ref communicator_exception representing the given @p error_code.
         * @param[in] comm the MPI communicator in which the error occurred
         * @param[in] error_code the occurred MPI error code
         */
        communicator_exception(const MPI_Comm& comm, const int error_code);

        /**
         * @brief Returns the MPI error message associated with the error code.
         * @return the error message (`[[nodiscard]]`)
         */
        [[nodiscard]]
        const char* what() const noexcept override { return error_msg_; }

        /**
         * @brief Returns the MPI rank on which the error occurred.
         * @return the MPI rank (`[[nodiscard]]`)
         */
        [[nodiscard]]
        int rank() const noexcept { return comm_rank_; }
        /**
         * @brief Returns the MPI error code.
         * @return the error code (`[[nodiscard]]`)
         */
        [[nodiscard]]
        int error_code() const noexcept { return error_code_; }

    private:
        const int error_code_;
        int comm_rank_;
        char error_msg_[MPI_MAX_ERROR_STRING];
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_COMMUNICATOR_EXCEPTION_HPP
