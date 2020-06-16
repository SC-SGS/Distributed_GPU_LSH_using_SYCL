/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-16
 *
 * @brief Implements a custom file exception class, which retrieves its what message from the MPI error code.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_FILE_EXCEPTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_FILE_EXCEPTION_HPP


#include <exception>
#include <iostream>

#include <mpi.h>


/**
 * @brief A custom file exception class wrapping a MPI error code.
 */
class mpi_file_exception : public std::exception {
public:
    /**
     * @brief Constructs a new @ref mpi_file_exception class.
     * @param[in] error_code the occurred MPI error code
     */
    mpi_file_exception(MPI_File&, const int error_code) : error_code_(error_code) {
        // retrieve error message
        int error_msg_length;
        MPI_Error_string(error_code, error_msg_, &error_msg_length);
        error_msg_[error_msg_length] = '\0';
    }

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
    /// The MPI error code that occurred.
    const int error_code_;
    /// The MPI error string associated with @ref error_code_.
    char error_msg_[MPI_MAX_ERROR_STRING];
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_FILE_EXCEPTION_HPP
