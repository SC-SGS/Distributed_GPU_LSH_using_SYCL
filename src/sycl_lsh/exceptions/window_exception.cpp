/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-21
 */

#include <sycl_lsh/exceptions/window_exception.hpp>


sycl_lsh::mpi::window_exception::window_exception(const MPI_Win&, const int error_code) : error_code_(error_code) {
    // retrieve error message
    int error_msg_length;
    MPI_Error_string(error_code, error_msg_, &error_msg_length);
    error_msg_[error_msg_length] = '\0';
}