/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-18
 */

#include <sycl_lsh/exceptions/communicator_exception.hpp>


sycl_lsh::communicator_exception::communicator_exception(const MPI_Comm& comm, const int error_code) : error_code_(error_code) {
    // get current MPI rank
    MPI_Comm_rank(comm, &comm_rank_);

    // retrieve error message
    int error_msg_length;
    MPI_Error_string(error_code, error_msg_, &error_msg_length);
    error_msg_[error_msg_length] = '\0';
}