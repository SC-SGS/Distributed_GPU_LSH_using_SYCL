/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements report/evaluation functions for nearest-neighbor results.
 */

#ifndef SYCL_LSH_NEAREST_NEIGHBORS_REPORT_HPP
#define SYCL_LSH_NEAREST_NEIGHBORS_REPORT_HPP
#pragma once

#include "sycl_lsh/constants.hpp"         // sycl_lsh::index_type, sycl_lsh::real_type
#include "sycl_lsh/matrix.hpp"            // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator

#include "mpi/detail/logging.hpp"

#include <tuple>  // std::tuple, std::make_tuple

namespace sycl_lsh::report {

/**
 * @brief Calculates the recall using: \f$ \frac{true\ positives}{relevant\ elements} \f$
 * @param[in] calculated_indices the calculated nearest-neighbor indices
 * @param[in] correct_indices the correct nearest-neighbor indices
 * @param[in] comm the used mpi::communicator
 * @return the resulting recall (`[[nodiscard]]`)
 *
 * @throws sycl_lsh::exception if the required command line argument `evaluate_knn_file` isn't present in @p parser.
 * @throws sycl_lsh::exception if the parsed total number of points doesn't match with the current `total_size`.
 * @throws sycl_lsh::exception if the parsed number of points per MPI rank doesn't match with the current `rank_size`.
 * @throws sycl_lsh::exception if the parsed number of dimensions doesn't match with the current `dims`.
 */
[[nodiscard]] real_type recall(const aos_matrix<index_type> &calculated_indices, const aos_matrix<index_type> &correct_indices, const mpi::communicator &comm);

/**
 * @brief Calculates the error ratio using: \f$ \frac{1}{N} \cdot \sum\limits_{i = 0}^N (\frac{1}{k} \cdot \sum\limits_{j = 0}^k \frac{dist_{LSH_j}}{dist_{correct_j}}) \f$
 * @param[in] calculated_distances the calculated nearest-neighbor distances
 * @param[in] correct_distances the correct nearest-neighbor distances
 * @param[in] comm the used mpi::communicator
 * @return a `std::tuple` containing the resulting error ratio, the number of points for which no k nearest-neighbors could be
 *         found and the total number of nearest-neighbors that couldn't be found (`[[nodiscard]]`)
 *
 * @throws sycl_lsh::exception if the required command line argument `evaluate_knn_dist_file` isn't present in @p parser.
 * @throws sycl_lsh::exception if the parsed total number of points doesn't match with the current `total_size`.
 * @throws sycl_lsh::exception if the parsed number of points per MPI rank doesn't match with the current `rank_size`.
 * @throws sycl_lsh::exception if the parsed number of dimensions doesn't match with the current `dims`.
 */
[[nodiscard]] std::tuple<real_type, index_type, index_type> error_ratio(const aos_matrix<real_type> &calculated_distances, const aos_matrix<real_type> &correct_distances, const mpi::communicator &comm);

}  // namespace sycl_lsh::report

#endif  // SYCL_LSH_NEAREST_NEIGHBORS_REPORT_HPP
