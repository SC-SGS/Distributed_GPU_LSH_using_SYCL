/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-19
 */

#include <sycl_lsh/exceptions/file_exception.hpp>


sycl_lsh::mpi::file_exception::file_exception(const MPI_File& file, const int error_code) : error_code_(error_code) {
    // try retrieving the filename
    MPI_Info info;
    MPI_File_get_info(file, &info);
    int valuelen, flag;
    MPI_Info_get_valuelen(info, "filename", &valuelen, &flag);
    if (static_cast<bool>(flag)) {
        MPI_Info_get(info, "filename", valuelen, filename_, &flag);
        filename_[valuelen] = '\0';
    }

    // retrieve error message
    int error_msg_length;
    MPI_Error_string(error_code, error_msg_, &error_msg_length);
    error_msg_[error_msg_length] = '\0';
}