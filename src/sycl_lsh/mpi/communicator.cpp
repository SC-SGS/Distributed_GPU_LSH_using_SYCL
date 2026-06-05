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

std::vector<std::string> communicator::gather(const std::string &str) const {
    // if we have only a single MPI rank, just return it without any MPI calls
    if (this->size() == 1) {
        return { str };
    }

    // gather string size information
    const std::vector<int> sizes = this->gather(static_cast<int>(str.size()));

    // calculate displacements and create receive-buffer (on main rank only!)
    std::vector<char> recv_buffer{};
    std::vector<int> displacements(sizes.size());
    if (this->is_main_rank()) {
        int total_size{};
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            displacements[i] = total_size;
            total_size += sizes[i];
        }
        recv_buffer.resize(total_size);
    }

    // gather the strings on the MPI main rank
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Gatherv(str.data(), str.size(), detail::mpi_datatype<char>(), recv_buffer.data(), sizes.data(), displacements.data(), detail::mpi_datatype<char>(), communicator::main_rank(), comm_));

    // unpack the receive-buffer to the separate strings
    std::vector<std::string> result(sizes.size());
    if (this->is_main_rank()) {
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            result[i] = std::string(recv_buffer.begin() + displacements[i],
                                    recv_buffer.begin() + displacements[i] + sizes[i]);
        }
    }
    return result;
}

}  // namespace sycl_lsh::mpi
