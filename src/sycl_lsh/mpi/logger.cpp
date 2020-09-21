/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-21
 */

#include <sycl_lsh/mpi/logger.hpp>

#include <mpi.h>

#include <vector>
#include <numeric>


// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::logger::logger(const sycl_lsh::communicator& comm, std::ostream& out) : comm_(comm), out_(out) { }


// ---------------------------------------------------------------------------------------------------------- //
//                                                  logging                                                   //
// ---------------------------------------------------------------------------------------------------------- //
void sycl_lsh::logger::log(const int comm_rank, const std::string_view msg) {
    // print message only on requested MPI rank
    if (comm_rank == comm_.rank()) {
        out_ << msg;
    }
}

void sycl_lsh::logger::log(const std::string_view msg) {
    // print message only in master rank (MPI rank 0)
    log(0, msg);
}

void sycl_lsh::logger::log_on_master(const std::string_view msg) {
    // print message only in master rank (MPI rank 0)
    log(0, msg);
}

void sycl_lsh::logger::log_on_all(const std::string_view msg) {
    // get the sizes of each message
    std::vector<int> sizes(comm_.size());
    int msg_size = msg.size();
    MPI_Gather(&msg_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, comm_.get());

    // calculate total msg size
    int total_msg_size = std::accumulate(sizes.begin(), sizes.end(), 0);

    // calculate displacements
    std::vector<int> displacements(sizes.size(), 0);
    for (std::size_t i = 1; i < displacements.size(); ++i) {
        displacements[i] = displacements[i - 1] + sizes[i - 1];
    }

    // get all messages
    std::string total_msg(total_msg_size, ' ');
    MPI_Gatherv(msg.data(), msg.size(), MPI_CHAR, total_msg.data(), sizes.data(), displacements.data(), MPI_CHAR, 0, comm_.get());

    // print total msg on master rank
    if (comm_.rank() == 0) {
        out_ << total_msg;
    }
}