/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements a distributed sort algorithm using MPI.
 */

#ifndef SYCL_LSH_MPI_DETAIL_SORT_HPP
#define SYCL_LSH_MPI_DETAIL_SORT_HPP
#pragma once

#include "sycl_lsh/mpi/communicator.hpp"      // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/type_cast.hpp"  // sycl_lsh::mpi::detail::mpi_datatype
#include "sycl_lsh/mpi/detail/utility.hpp"    // SYCL_LSH_MPI_ERROR_CHECK

#include "mpi.h"  // MPI_Send, MPI_Recv, MPI_STATUS_IGNORE

#include <algorithm>  // std::copy, std::sort
#include <cstddef>    // std::size_t
#include <vector>     // std::vector

namespace sycl_lsh::mpi::detail {

/**
 * @brief Sort a part of the overall data.
 * @tparam real_type the used floating point type
 * @param[in] data the data to sort
 * @param[in] send_rank the MPI rank to send the data to
 * @param[in] recv_rank the MPI rank to receive the data from
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 */
template <typename real_type>
void pairwise_exchange(std::vector<real_type> &data, const int send_rank, const int recv_rank, const communicator &comm) {
    std::vector<real_type> all(2 * data.size());
    constexpr int merge_tag = 1;
    constexpr int sorted_tag = 2;

    if (comm.rank() == send_rank) {
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Send(data.data(), data.size(), mpi_datatype<real_type>(), recv_rank, merge_tag, comm.get()));
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Recv(data.data(), data.size(), mpi_datatype<real_type>(), recv_rank, sorted_tag, comm.get(), MPI_STATUS_IGNORE));
    } else {
        std::copy(data.cbegin(), data.cend(), all.begin());
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Recv(all.data() + data.size(), data.size(), mpi_datatype<real_type>(), send_rank, merge_tag, comm.get(), MPI_STATUS_IGNORE));

        // perform local sort
        std::sort(all.begin(), all.end());

        const std::size_t their_start = send_rank > comm.rank() ? data.size() : 0;
        const std::size_t my_start = send_rank > comm.rank() ? 0 : data.size();
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Send(all.data() + their_start, data.size(), mpi_datatype<real_type>(), send_rank, sorted_tag, comm.get()));
        std::copy(all.cbegin() + my_start, all.cbegin() + my_start + data.size(), data.begin());
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
void sort(std::vector<real_type> &data, const communicator &comm) {
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

}  // namespace sycl_lsh::mpi::detail

#endif  // SYCL_LSH_INCLUDE_MPI_DETAIL_SORT_HPP
