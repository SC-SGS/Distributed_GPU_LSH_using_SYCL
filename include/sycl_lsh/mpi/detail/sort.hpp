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

#include <algorithm>    // std::copy, std::sort
#include <cstddef>      // std::size_t
#include <iterator>     // std::distance, std::random_access_iterator_tag, std::iterator_traits
#include <type_traits>  // std::is_base_of_v
#include <vector>       // std::vector

namespace sycl_lsh::mpi::detail {

/**
 * @brief Sort a part of the overall data.
 * @tparam Iterator the iterator type to the underlying data
 * @param[in] begin the iterator pointing to the start of the range
 * @param[in] end the iterator pointing to the end of the range
 * @param[in] send_rank the MPI rank to send the data to
 * @param[in] recv_rank the MPI rank to receive the data from
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 */
template <typename Iterator>
void pairwise_exchange(Iterator begin, Iterator end, const int send_rank, const int recv_rank, const communicator &comm) {
    // only random access iterators allowed
    // NOTE: actually, only contiguous iterators are allowed, bit this category is C++20 only
    static_assert(std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<Iterator>::iterator_category>, "Unsupported Iterator type!");

    const std::size_t size = std::distance(begin, end);

    std::vector<real_type> all(2 * size);
    constexpr int merge_tag = 1;
    constexpr int sorted_tag = 2;

    if (comm.rank() == send_rank) {
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Send(&(*begin), size, mpi_datatype<real_type>(), recv_rank, merge_tag, comm));
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Recv(&(*begin), size, mpi_datatype<real_type>(), recv_rank, sorted_tag, comm, MPI_STATUS_IGNORE));
    } else {
        std::copy(begin, end, all.begin());
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Recv(all.data() + size, size, mpi_datatype<real_type>(), send_rank, merge_tag, comm, MPI_STATUS_IGNORE));

        // perform local sort
        std::sort(all.begin(), all.end());

        const std::size_t their_start = send_rank > comm.rank() ? size : 0;
        const std::size_t my_start = send_rank > comm.rank() ? 0 : size;
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Send(all.data() + their_start, size, mpi_datatype<real_type>(), send_rank, sorted_tag, comm));
        std::copy(all.cbegin() + my_start, all.cbegin() + my_start + size, begin);
    }
}

// https://stackoverflow.com/questions/23633916/how-does-mpi-odd-even-sort-work
/**
 * @brief Implements a distributed odd-even sort using MPI.
 * @tparam Iterator the iterator type to the underlying data
 * @param[in] begin the iterator pointing to the start of the range
 * @param[in] end the iterator pointing to the end of the range
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 */
template <typename Iterator>
void sort(Iterator begin, Iterator end, const communicator &comm) {
    // only random access iterators allowed
    // NOTE: actually, only contiguous iterators are allowed, bit this category is C++20 only
    static_assert(std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<Iterator>::iterator_category>, "Unsupported Iterator type!");

    // sort local vector
    std::sort(begin, end);

    // odd-even
    for (std::size_t i = 1; i <= static_cast<std::size_t>(comm.size()); ++i) {
        if ((i + comm.rank()) % 2 == 0) {
            if (comm.rank() < comm.size() - 1) {
                pairwise_exchange(begin, end, comm.rank(), comm.rank() + 1, comm);
            }
        } else if (comm.rank() > 0) {
            pairwise_exchange(begin, end, comm.rank() - 1, comm.rank(), comm);
        }
    }
}

}  // namespace sycl_lsh::mpi::detail

#endif  // SYCL_LSH_INCLUDE_MPI_DETAIL_SORT_HPP
