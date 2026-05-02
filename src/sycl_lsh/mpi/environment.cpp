/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/mpi/environment.hpp"

#include "sycl_lsh/mpi/detail/utility.hpp"  // SYCL_LSH_MPI_ERROR_CHECK

#include "mpi.h"  // MPI_Initialized, MPI_Finalized

namespace sycl_lsh::mpi {

bool is_initialized() {
    int flag{};
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Initialized(&flag));
    return static_cast<bool>(flag);
}

bool is_finalized() {
    int flag{};
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Finalized(&flag));
    return static_cast<bool>(flag);
}

bool is_active() {
    return is_initialized() && !is_finalized();
}

}  // namespace sycl_lsh::mpi