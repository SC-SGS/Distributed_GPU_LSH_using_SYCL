/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-22
 *
 * @brief Exception class for errors occurring in MPI windows.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_WINDOW_EXCEPTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_WINDOW_EXCEPTION_HPP

#include <exception>

#include <mpi.h>

namespace sycl_lsh::mpi {

    /**
     * @brief Exception class for errors occurring in MPI windows.
     */
    class window_exception : public std::exception {
    public:
        /**
         * @brief Constructs a new @ref window_exception representing the given @p error_code.
         * @param[in] win the MPI window in which the error occurred
         * @param[in] error_code the occurred MPI error code
         */
        window_exception(const MPI_Win& win, int error_code);

        /**
         * @brief Returns the MPI error message associated with the error code.
         * @return the error message (`[[nodiscard]]`)
         */
        [[nodiscard]]
        const char* what() const noexcept override { return error_msg_; }

        /**
         * @brief Returns the MPI error code.
         * @return the error code (`[[nodiscard]]`)
         */
        [[nodiscard]]
        int error_code() const noexcept { return error_code_; }

    private:
        const int error_code_;
        char error_msg_[MPI_MAX_ERROR_STRING];
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_WINDOW_EXCEPTION_HPP
