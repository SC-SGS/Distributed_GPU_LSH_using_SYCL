/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Defines some utility functions for the MPI usage.
 */

#ifndef SYCL_LSH_MPI_DETAIL_UTILITY_HPP
#define SYCL_LSH_MPI_DETAIL_UTILITY_HPP
#pragma once

#include <string>  // std::string

/**
 * @def SYCL_LSH_MPI_ERROR_CHECK
 * @brief Check the MPI error @p err. If @p err signals an error, throw a @ref sycl_lsh::mpi_exception.
 *
 * @throws sycl_lsh::mpi_exception if the error code signals a failure
 */
#if defined(SYCL_LSH_ASSERTS_ENABLED)
    #define SYCL_LSH_MPI_ERROR_CHECK(err) sycl_lsh::mpi::detail::mpi_error_check(err)
#else
    #define SYCL_LSH_MPI_ERROR_CHECK(...) __VA_ARGS__
#endif

namespace sycl_lsh::mpi::detail {

/**
 * @brief Checks whether @p err is equal to MPI_SUCCESS. If this is not the case, throws an exception.
 * @param[in] err the error code to check
 * @throws sycl_lsh::mpi_exception if the error code signals a failure
 */
void mpi_error_check(int err);

/**
 * @brief Convert the @p error_code to the respective MPI error message.
 * @param[in] error_code the MPI related error code
 * @return the associated MPI error message (`[[nodiscard]]`)
 */
[[nodiscard]] std::string mpi_error_code_to_string(int error_code);

}  // namespace sycl_lsh::mpi::detail

#endif  // SYCL_LSH_MPI_DETAIL_UTILITY_HPP