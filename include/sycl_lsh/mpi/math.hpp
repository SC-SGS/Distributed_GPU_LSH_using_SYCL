/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 *
 * @brief Implements needed math function using MPI.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MATH_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MATH_HPP

#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/type_cast.hpp>

#include <mpi.h>

namespace sycl_lsh::mpi {

    /**
     * @brief Sums the given @p value over all MPI ranks.
     * @details Returns the result on **all** MPI ranks.
     * @tparam T the type of the @p value to sum
     * @param[in] value the value to sum
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @return the resulting sum
     */
    template <typename T>
    [[nodiscard]]
    inline T sum(T value, const communicator& comm) {
        T sums = 0.0;
        MPI_Allreduce(&value, &sums, 1, type_cast<T>(), MPI_SUM, comm.get());
        return sums;
    }

    /**
     * @brief Averages the given @p value over all MPI ranks.
     * @details Returns the result on **all** MPI ranks.
     * @tparam T the type of the @p value to average
     * @param[in] value the value to average
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @return the resulting average
     */
    template <typename T>
    [[nodiscard]]
    inline T average(T value, const communicator& comm) {
        T sums = sum(value, comm);
        return sums / comm.size();
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MATH_HPP
