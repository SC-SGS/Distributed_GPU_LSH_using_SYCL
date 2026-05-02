/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/mpi/detail/utility.hpp"

#include "sycl_lsh/exceptions/exceptions.hpp"  // sycl_lsh::mpi_exception

#include "fmt/format.h"  // fmt::format
#include "mpi.h"         // MPI_Get_processor_name

#include <string>  // std::string

namespace sycl_lsh::mpi::detail {

void mpi_error_check(const int err) {
    if (err != MPI_SUCCESS) {
        const std::string err_msg = mpi_error_code_to_string(err);
        if (!err_msg.empty()) {
            throw mpi_exception{ fmt::format("MPI error {}: {}", err, err_msg) };
        } else {
            throw mpi_exception{ fmt::format("MPI error {}", err) };
        }
    }
}

std::string mpi_error_code_to_string(const int error_code) {
    std::string err_str(MPI_MAX_ERROR_STRING, '\0');
    int err_str_len{};
    const int res = MPI_Error_string(error_code, err_str.data(), &err_str_len);
    if (res == MPI_SUCCESS) {
        return err_str.substr(0, err_str.find_first_of('\0'));
    }
    return std::string{};
}

}  // namespace sycl_lsh::mpi::detail