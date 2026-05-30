/**
* @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator

#include "sycl_lsh/mpi/detail/utility.hpp"  // SYCL_LSH_MPI_ERROR_CHECK

#include "mpi/mpi.h"  // MPI_Comm, MPI_Comm_size, MPI_Comm_rank, MPI_Barrier

namespace sycl_lsh::mpi {

communicator::communicator(MPI_Comm comm) noexcept :
    comm_{ comm } { }

int communicator::size() const {
    int comm_size{};
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Comm_size(comm_, &comm_size));
    return comm_size;
}

int communicator::rank() const {
    int comm_rank{};
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Comm_rank(comm_, &comm_rank));
    return comm_rank;
}

bool communicator::is_main_rank() const {
    return this->rank() == main_rank();
}

void communicator::barrier() const {
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Barrier(comm_));
}

}  // namespace sycl_lsh::mpi
