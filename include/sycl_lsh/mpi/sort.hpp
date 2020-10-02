/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-02
 *
 * @brief Implements a distributed sort algorithm using MPI.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SORT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SORT_HPP

#include <sycl_lsh/mpi/communicator.hpp>

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace sycl_lsh::mpi {


    template <typename real_type>
    void pairwise_exchange(std::vector<real_type>& a, const int sendrank, const int recvrank, const communicator& comm) {
        std::vector<real_type> remote(a.size());
        std::vector<real_type> all(2 * a.size());
        constexpr int merge_tag = 1;
        constexpr int sorted_tag = 2;

        if (comm.rank() == sendrank) {
            MPI_Send(a.data(), a.size(), type_cast<real_type>(), recvrank, merge_tag, comm.get());
            MPI_Recv(a.data(), a.size(), type_cast<real_type>(), recvrank, sorted_tag, comm.get(), MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(remote.data(), remote.size(), type_cast<real_type>(), sendrank, merge_tag, comm.get(), MPI_STATUS_IGNORE);
            std::copy(a.begin(), a.end(), all.begin());
            std::copy(remote.begin(), remote.end(), all.begin() + a.size());

            std::sort(all.begin(), all.end());

            std::size_t theirstart = sendrank > comm.rank() ? a.size() : 0;
            std::size_t mystart = sendrank > comm.rank() ? 0 : a.size();
            MPI_Send(all.data() + theirstart, a.size(), type_cast<real_type>(), sendrank, sorted_tag, comm.get());
            std::copy(all.begin() + mystart, all.begin() + mystart + a.size(), a.begin());
        }
    }

    // https://stackoverflow.com/questions/23633916/how-does-mpi-odd-even-sort-work
    template <typename real_type>
    void odd_even_sort(std::vector<real_type>& a, const communicator& comm) {
        // sort local vector
        std::sort(a.begin(), a.end());

        // odd-even
        for (std::size_t i = 1; i <= static_cast<std::size_t>(comm.size()); ++i) {
            if ((i + comm.rank()) % 2 == 0) {
                if (comm.rank() < comm.size() - 1) {
                    pairwise_exchange(a, comm.rank(), comm.rank() + 1, comm);
                }
            } else if (comm.rank() > 0) {
                pairwise_exchange(a, comm.rank() - 1, comm.rank(), comm);
            }
        }
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SORT_HPP
