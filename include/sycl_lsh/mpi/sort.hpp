/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 *
 * @brief Implements a distributed sort algorithm using MPI.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SORT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SORT_HPP

#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/type_cast.hpp>

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace sycl_lsh::mpi {

    /**
     * @brief Sort a part of the overall data.
     * @tparam real_type the used floating point type
     * @param[in] data the data to sort
     * @param[in] sendrank the MPI rank to send the data to
     * @param[in] recvrank the MPI rank to receive the data from
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     */
    template <typename real_type>
    inline void pairwise_exchange(std::vector<real_type>& data, const int sendrank, const int recvrank, const communicator& comm) {
        std::vector<real_type> all(2 * data.size());
        constexpr int merge_tag = 1;
        constexpr int sorted_tag = 2;

        if (comm.rank() == sendrank) {
            MPI_Send(data.data(), data.size(), type_cast<real_type>(), recvrank, merge_tag, comm.get());
            MPI_Recv(data.data(), data.size(), type_cast<real_type>(), recvrank, sorted_tag, comm.get(), MPI_STATUS_IGNORE);
        } else {
            std::copy(data.begin(), data.end(), all.begin());
            MPI_Recv(all.data() + data.size(), data.size(), type_cast<real_type>(), sendrank, merge_tag, comm.get(), MPI_STATUS_IGNORE);

            std::sort(all.begin(), all.end());

            std::size_t theirstart = sendrank > comm.rank() ? data.size() : 0;
            std::size_t mystart = sendrank > comm.rank() ? 0 : data.size();
            MPI_Send(all.data() + theirstart, data.size(), type_cast<real_type>(), sendrank, sorted_tag, comm.get());
            std::copy(all.begin() + mystart, all.begin() + mystart + data.size(), data.begin());
        }
    }

    // https://stackoverflow.com/questions/23633916/how-does-mpi-odd-even-sort-work
    /**
     * @brief Implements a distributed odd-even sort using MPI.
     * @tparam real_type the used floating point type
     * @param[in] data the data to sort
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     */
    template <typename real_type>
    inline void sort(std::vector<real_type>& data, const communicator& comm) {
        // sort local vector
        std::sort(data.begin(), data.end());

        // odd-even
        for (std::size_t i = 1; i <= static_cast<std::size_t>(comm.size()); ++i) {
            if ((i + comm.rank()) % 2 == 0) {
                if (comm.rank() < comm.size() - 1) {
                    pairwise_exchange(data, comm.rank(), comm.rank() + 1, comm);
                }
            } else if (comm.rank() > 0) {
                pairwise_exchange(data, comm.rank() - 1, comm.rank(), comm);
            }
        }
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SORT_HPP
