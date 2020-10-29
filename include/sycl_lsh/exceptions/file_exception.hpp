/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-19
 *
 * @brief Exception class for errors occurring in MPI files.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_EXCEPTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_EXCEPTION_HPP

#include <exception>

#include <mpi.h>

namespace sycl_lsh::mpi {

    /**
     * @brief Exception class for errors occurring in MPI files.
     */
    class file_exception : public std::exception {
    public:
        /**
         * @brief Constructs a new @ref file_exception representing the given @p error_code.
         * @param[in] file the MPI file in which the error occurred
         * @param[in] error_code the occurred MPI error code
         */
        file_exception(const MPI_File& file, int error_code);

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
        /**
         * @brief Returns the filename of the file on which the MPI error occurred.
         * @details Returns `"unknown"` if retrieving the filename isn't supported.
         * @return the filename (`[[nodiscard]]`)
         */
        [[nodiscard]]
        const char* filename() const noexcept { return filename_; }

    private:
        const int error_code_;
        char filename_[MPI_MAX_INFO_VAL] = "unknown";
        char error_msg_[MPI_MAX_ERROR_STRING];
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FILE_EXCEPTION_HPP
