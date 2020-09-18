/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-18
 */

#include <sycl_lsh/exceptions/file_exception.hpp>


sycl_lsh::file_exception::file_exception(const MPI_File& file, const int error_code) : error_code_(error_code) {
    // retrieve error message
    int error_msg_length;
    MPI_Error_string(error_code, error_msg_, &error_msg_length);
    error_msg_[error_msg_length] = '\0';
}