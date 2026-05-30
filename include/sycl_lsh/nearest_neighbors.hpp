/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the @ref sycl_lsh::knn class representing the result of the k-nearest-neighbor search.
 */

#ifndef SYCL_LSH_KNN_HPP
#define SYCL_LSH_KNN_HPP
#pragma once

#include "sycl_lsh/data_attributes.hpp"   // sycl_lsh::data_attributes
#include "sycl_lsh/data_set.hpp"          // sycl_lsh::data_set
#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/logger.hpp"        // sycl_lsh::mpi::logger
#include "sycl_lsh/options.hpp"           // sycl_lsh::options

#include "sycl_lsh/matrix.hpp"  // sycl_lsh::aos_matrix
#include <tuple>   // std::tuple, std::make_tuple
#include <vector>  // std::vector

namespace sycl_lsh {

/**
 * @brief Class representing the result of the k-nearest-neighbor search.
 */
class nearest_neighbors {
  public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::knn object given @p opt.k, the number of nearest-neighbors to search for.
     * @param[in] opt the provided options
     * @param[in] data the used @ref sycl_lsh::data representing the used data set
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     *
     * @pre @p opt.k **must** be greater than `0`.
     */
    nearest_neighbors(const options &opt, const data_set &data, const mpi::communicator &comm, const mpi::logger &logger);
    /**
     * @brief Construct a new @ref sycl_lsh::knn object given @p k, the number of nearest-neighbors to search for.
     * @param[in] k the number of nearest-neighbors to search for
     * @param[in] data the used @ref sycl_lsh::data representing the used data set
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     *
     * @pre @p k **must** be greater than `0`.
     */
    nearest_neighbors(index_type k, const data_set &data, const mpi::communicator &comm, const mpi::logger &logger);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                knn results                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Returns the IDs (indices) of the k-nearest-neighbors found for @p point.
     * @param[in] point the data point to return the nearest-neighbors for
     * @return the indices of the found k-nearest-neighbors of @p point (`[[nodiscard]]`)
     *
     * @attention Copies the IDs to the result vector!
     *
     * @pre @p point must be in the range `[0, number of data points on the current MPI rank)`.
     */
    [[nodiscard]] std::vector<index_type> get_knn_ids(index_type point) const;
    /**
     * @brief Returns the distances of the k-nearest-neighbors found for @p point.
     * @param[in] point the data point to return the nearest-neighbors for
     * @return the distances of the found k-nearest-neighbors of @p point (`[[nodiscard]]`)
     *
     * @attention Copies the distances to the result vector!
     *
     * @pre @p point must be in the range `[0, number of data points on the current MPI rank)`.
     */
    [[nodiscard]] std::vector<real_type> get_knn_dists(index_type point) const;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                  save knn                                                  //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Saves the calculated k-nearest-neighbor IDs. \n
     *        **Always** saves the k-nearest-neighbor IDs in *Array of Structs* layout.
     * @param[in] opt the used @ref sycl_lsh::options
     *
     * @throws sycl_lsh::exception if the command line argument `knn_save_file` isn't present in @p parser.
     */
    void save_knns(const options &opt);
    /**
     * @brief Saves the calculated k-nearest-neighbor distances. \n
     *        **Always** saves the k-nearest-neighbor distances in *Array of Structs* layout.
     * @param[in] opt the used @ref sycl_lsh::options
     *
     * @throws sycl_lsh::exception if the command line argument `knn_dist_save_file` isn't present in @p parser.
     */
    void save_distances(const options &opt);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                evaluate knn                                                //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Calculates the recall using: \f$ \frac{true\ positives}{relevant\ elements} \f$
     * @param[in] opt the used @ref sycl_lsh::options
     * @return the resulting recall (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if the required command line argument `evaluate_knn_file` isn't present in @p parser.
     * @throws sycl_lsh::exception if the parsed total number of points doesn't match with the current `total_size`.
     * @throws sycl_lsh::exception if the parsed number of points per MPI rank doesn't match with the current `rank_size`.
     * @throws sycl_lsh::exception if the parsed number of dimensions doesn't match with the current `dims`.
     */
    [[nodiscard]] real_type recall(const options &opt);
    /**
     * @brief Calculates the error ratio using: \f$ \frac{1}{N} \cdot \sum\limits_{i = 0}^N (\frac{1}{k} \cdot \sum\limits_{j = 0}^k \frac{dist_{LSH_j}}{dist_{correct_j}}) \f$
     * @param[in] opt the used @ref sycl_lsh::options
     * @return a `std::tuple` containing the resulting error ratio, the number of points for which no k k-nearest-neighbors could be
     *         found and the total number of nearest-neighbors that couldn't be found (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if the required command line argument `evaluate_knn_dist_file` isn't present in @p parser.
     * @throws sycl_lsh::exception if the parsed total number of points doesn't match with the current `total_size`.
     * @throws sycl_lsh::exception if the parsed number of points per MPI rank doesn't match with the current `rank_size`.
     * @throws sycl_lsh::exception if the parsed number of dimensions doesn't match with the current `dims`.
     */
    [[nodiscard]] std::tuple<real_type, index_type, index_type> error_ratio(const options &opt);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                   getter                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Returns the host buffer containing the k-nearest-neighbor IDs used to hide the MPI communication.
     * @return the knn host buffer (`[[nodiscard]]`)
     */
    [[nodiscard]] aos_matrix<index_type> &get_knn_indices() noexcept { return knn_indices_; }

    /**
     * @brief Returns the host buffer containing the k-nearest-neighbor distances used to hide the MPI communication.
     * @details The distances are calculated without the use of `std::sqrt`!
     * @return the knn distances host buffer (`[[nodiscard]]`)
     */
    [[nodiscard]] aos_matrix<real_type> &get_knn_distances() noexcept { return knn_distances_; }

  private:
    /// The data attributes.
    data_attributes attr_;
    /// The associated MPI communicator.
    mpi::communicator comm_;
    /// The associated MPI logger.
    const mpi::logger &logger_;

    /// The number of nearest-neighbors to calculate.
    index_type k_;

    /// The host data for the nearest-neighbors.
    aos_matrix<index_type> knn_indices_;
    /// The host data for the nearest-neighbor distances.
    aos_matrix<real_type> knn_distances_;
};

}  // namespace sycl_lsh

#endif  // SYCL_LSH_KNN_HPP
