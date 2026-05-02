/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements needed math function using MPI.
 */

#ifndef SYCL_LSH_MPI_DETAIL_MATH_HPP
#define SYCL_LSH_MPI_DETAIL_MATH_HPP
#pragma once

#include "sycl_lsh/mpi/communicator.hpp"      // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/type_cast.hpp"  // sycl_lsh::mpi::detail::mpi_datatype
#include "sycl_lsh/mpi/detail/utility.hpp"    // SYCL_LSH_MPI_ERROR_CHECK

#include "mpi.h"  // MPI_Allreduce, MPI_SUM

namespace sycl_lsh::mpi::detail {

/**
 * @brief Sums the given @p value over all MPI ranks.
 * @details Returns the result on **all** MPI ranks.
 * @tparam T the type of the @p value to sum
 * @param[in] value the value to sum
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 * @return the resulting sum (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] T sum(const T value, const communicator &comm) {
    T sum{};
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Allreduce(&value, &sum, 1, mpi_datatype<T>(), MPI_SUM, comm.get()));
    return sum;
}

/**
 * @brief Averages the given @p value over all MPI ranks.
 * @details Returns the result on **all** MPI ranks.
 * @tparam T the type of the @p value to average
 * @param[in] value the value to average
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 * @return the resulting average (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] T average(const T value, const communicator &comm) {
    const T sum = sum(value, comm);
    return sum / comm.size();
}

}  // namespace sycl_lsh::mpi::detail

#endif  // SYCL_LSH_MPI_DETAIL_MATH_HPP
