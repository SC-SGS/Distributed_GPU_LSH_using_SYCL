/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-22
 */

#include <sycl_lsh/mpi/logger.hpp>
#include <sycl_lsh/mpi/type_cast.hpp>

#include <mpi.h>

#include <vector>
#include <numeric>


// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::mpi::logger::logger(const sycl_lsh::mpi::communicator& comm, std::ostream& out) : comm_(comm), out_(out) { }


// ---------------------------------------------------------------------------------------------------------- //
//                                                  logging                                                   //
// ---------------------------------------------------------------------------------------------------------- //
void sycl_lsh::mpi::logger::log(const int comm_rank, const std::string_view msg) {
    // print message only on requested MPI rank
    if (comm_rank == comm_.rank()) {
        out_ << msg;
    }
}

void sycl_lsh::mpi::logger::log(const std::string_view msg) {
    // print message only in master rank (MPI rank 0)
    log(0, msg);
}

void sycl_lsh::mpi::logger::log_on_master(const std::string_view msg) {
    // print message only in master rank (MPI rank 0)
    log(0, msg);
}

void sycl_lsh::mpi::logger::log_on_all(const std::string_view msg) {
    // get the sizes of each message
    std::vector<int> sizes(comm_.size());
    int msg_size = msg.size();
    MPI_Gather(&msg_size, 1, type_cast<typename decltype(sizes)::value_type>(), sizes.data(), 1, type_cast<decltype(msg_size)>(), 0, comm_.get());

    // calculate total msg size
    int total_msg_size = std::accumulate(sizes.begin(), sizes.end(), 0);

    // calculate displacements
    std::vector<int> displacements(sizes.size(), 0);
    for (std::size_t i = 1; i < displacements.size(); ++i) {
        displacements[i] = displacements[i - 1] + sizes[i - 1];
    }

    // get all messages
    std::string total_msg(total_msg_size, ' ');
    MPI_Gatherv(msg.data(), msg.size(), type_cast<char>(), total_msg.data(), sizes.data(), displacements.data(), type_cast<char>(), 0, comm_.get());

    // print total msg on master rank
    if (comm_.master_rank()) {
        out_ << total_msg;
    }
}