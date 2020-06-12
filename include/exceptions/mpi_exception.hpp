/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-12
 *
 * @brief Implements a custom exception class, which retrieves its what message from the MPI error code.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_EXCEPTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_EXCEPTION_HPP


#include <exception>

#include <mpi.h>


/**
 * @brief A custom exception class wrapping a MPI error code.
 */
class mpi_exception : public std::exception {
public:
    /**
     * @brief Constructs a new @ref mpi_exception class.
     * @param[in] communicator the communicator on which the error occurred
     * @param[in] error_code the occurred MPI error code
     */
    mpi_exception(const MPI_Comm& communicator, const int error_code) : error_code_(error_code) {
        // get current rank
        MPI_Comm_rank(communicator, &rank_);
        // retrieve error message
        int error_msg_length;
        MPI_Error_string(error_code, error_msg_, &error_msg_length);
        error_msg_[error_msg_length] = '\0';
    }

    /**
     * @brief Returns the MPI rank on which the error occurred.
     * @return the rank (`[[nodiscard]]`)
     */
    [[nodiscard]] int rank() const noexcept { return rank_; }
    /**
     * @brief Returns the MPI error code.
     * @return the error code (`[[nodiscard]]`)
     */
    [[nodiscard]] int error_code() const noexcept { return error_code_; }
    /**
     * @brief Returns the MPI error message associated with the error code.
     * @return the error message (`[[nodiscard]]`)
     */
    [[nodiscard]] const char* what() const noexcept override { return error_msg_; }

private:
    /// The MPI rank on which the error occurred.
    int rank_;
    /// The MPI error code that occurred.
    const int error_code_;
    /// The MPI error string associated with @ref error_code_.
    char error_msg_[MPI_MAX_ERROR_STRING];
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_EXCEPTION_HPP
