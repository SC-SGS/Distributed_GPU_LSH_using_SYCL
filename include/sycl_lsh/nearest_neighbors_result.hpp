/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements a wrapper class for the nearest-neighbor results.
 */

#ifndef SYCL_LSH_NEAREST_NEIGHBORS_RESULT_HPP
#define SYCL_LSH_NEAREST_NEIGHBORS_RESULT_HPP
#pragma once

#include "sycl_lsh/constants.hpp"              // sycl_lsh::index_type, sycl_lsh::real_type
#include "sycl_lsh/data_set.hpp"               // sycl_lsh::data_set
#include "sycl_lsh/matrix.hpp"                 // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"       // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/file_parser_types.hpp"  // sycl_lsh::mpi::file_parser_type
#include "sycl_lsh/profiler.hpp"               // sycl_lsh::profiler

#include <cstddef>   // std::size_t
#include <optional>  // std::optional
#include <string>    // std::string
#include <tuple>     // std::tuple, std::make_tuple
#include <vector>    // std::vector

namespace sycl_lsh {

/**
 * @brief Small wrapper containing the results of a nearest-neighbors search.
 */
class nearest_neighbors_result {
    // befriend nearest_neighbors class
    friend class nearest_neighbors;

  public:
    /**
     * @brief Return the number of nearest-neighbors that were calculated.
     * @return the number of nearest-neighbors (`[[nodiscard]]`)
     */
    [[nodiscard]] index_type get_n_neighbors() const noexcept { return indices_.num_cols(); }

    /**
     * @brief Check if reporting the calculated distances was enabled.
     * @return `true` if the distances are reported, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] bool has_distances() const noexcept { return distances_.has_value(); }

    /**
     * @brief Return all calculated indices on this MPI rank.
     * @return the calculated indices on this MPI rank (`[[nodiscard]]`)
     */
    [[nodiscard]] const aos_matrix<index_type> &indices() const noexcept { return indices_; }

    /**
     * @brief Return the indices of the point @p idx on this MPI rank.
     * @details Performs a dynamic memory allocation. Therefore, iterating over the columns in row @p idx in the return
     *         value of @ref sycl_lsh::nearest_neighbors_result::indices() may be more performant.
     * @param[in] idx the index of the point
     * @return the calculated nearest-neighbor indices of point @p idx (`[[nodiscard`]])
     *
     * @throws sycl_lsh::exception if @p idx is **not** in the range `[0, number of point on this MPI rank)`
     */
    [[nodiscard]] std::vector<index_type> indices(std::size_t idx) const;

    /**
     * @brief Return all calculated distances on this MPI rank if reporting them was enabled.
     * @return the calculated distances on this MPI rank, otherwise `std::nullopt` (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::optional<aos_matrix<real_type>> &distances() const noexcept { return distances_; }

    /**
     * @brief Return the distances of the point @p idx on this MPI rank if reporting them was enabled.
     * @details Performs a dynamic memory allocation. Therefore, iterating over the columns in row @p idx in the return
     *         value of @ref sycl_lsh::nearest_neighbors_result::distances() may be more performant.
     * @param[in] idx the index of the point
     * @return the calculated nearest-neighbor distances of point @p idx (`[[nodiscard`]])
     *
     * @throws sycl_lsh::exception if @p idx is **not** in the range `[0, number of point on this MPI rank)`
     * @throws sycl_lsh::exception if distances reporting was not enabled
     */
    [[nodiscard]] std::vector<real_type> distances(std::size_t idx) const;

    /**
     * @brief Save the calculated indices from **all** MPI ranks to the @p filename.
     * @param[in] filename the filename to write the indices to
     * @param[in] file_parser the @ref sycl_lsh::mpi::file_parser_type
     */
    void save_indices(const std::string &filename, mpi::file_parser_type file_parser = mpi::file_parser_type::binary) const;

    /**
     * @brief Save the calculated distances from **all** MPI ranks to the @p filename if reporting them was enabled.
     * @param[in] filename the filename to write the distances to
     * @param[in] file_parser the @ref sycl_lsh::mpi::file_parser_type
     *
     * @throws sycl_lsh::exception if distances reporting was not enabled
     */
    void save_distances(const std::string &filename, mpi::file_parser_type file_parser = mpi::file_parser_type::binary) const;

    /**
     * @brief Calculates the recall using: \f$ \frac{true\ positives}{relevant\ elements} \f$
     * @param[in] correct_indices the correct nearest-neighbor indices
     * @return the resulting recall (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if the shape of the @p correct_indices does not match with the shape of the calculated ones
     */
    [[nodiscard]] real_type recall(const aos_matrix<index_type> &correct_indices) const;
    /**
     * @brief Calculates the recall using: \f$ \frac{true\ positives}{relevant\ elements} \f$
     * @param[in] filename the filename storing the correct indices
     * @param[in] file_parser the @ref sycl_lsh::mpi::file_parser_type
     * @return the resulting recall (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if the shape of the indices read from @p filename does not match with the shape of the calculated ones
     */
    [[nodiscard]] real_type recall(const std::string &filename, mpi::file_parser_type file_parser = mpi::file_parser_type::binary) const;

    /**
     * @brief Calculates the error ratio using: \f$ \frac{1}{N} \cdot \sum\limits_{i = 0}^N (\frac{1}{k} \cdot \sum\limits_{j = 0}^k \frac{dist_{LSH_j}}{dist_{correct_j}}) \f$
     * @param[in] correct_distances the correct nearest-neighbor distances
     * @return a std::tuple containing the resulting error ratio, the number of points for which no k nearest-neighbors could be
     *         found and the total number of nearest-neighbors that couldn't be found (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if the shape of the @p correct_distances does not match with the shape of the calculated ones
     * @throws sycl_lsh::exception if distances reporting was not enabled
     */
    [[nodiscard]] std::tuple<real_type, index_type, index_type> error_ratio(const aos_matrix<real_type> &correct_distances) const;
    /**
     * @brief Calculates the error ratio using: \f$ \frac{1}{N} \cdot \sum\limits_{i = 0}^N (\frac{1}{k} \cdot \sum\limits_{j = 0}^k \frac{dist_{LSH_j}}{dist_{correct_j}}) \f$
     * @param[in] filename the filename storing the correct distances
     * @param[in] file_parser the @ref sycl_lsh::mpi::file_parser_type
     * @return a std::tuple containing the resulting error ratio, the number of points for which no k nearest-neighbors could be
     *         found and the total number of nearest-neighbors that couldn't be found (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if the shape of the distances read from @p filename does not match with the shape of the calculated ones
     * @throws sycl_lsh::exception if distances reporting was not enabled
     */
    [[nodiscard]] std::tuple<real_type, index_type, index_type> error_ratio(const std::string &filename, mpi::file_parser_type file_parser = mpi::file_parser_type::binary) const;

  private:
    /**
     * @brief Create a new nearest-neighbor result by only providing the calculated indices.
     * @param[in] comm the associated @ref sycl_lsh::mpi::communicator
     * @param[in] data the queried @ref sycl_lsh::data_set
     * @param[in] indices the calculated nearest-neighbor indices
     * @param[in] profiler the optional @ref sycl_lsh::profiler
     */
    explicit nearest_neighbors_result(mpi::communicator comm, data_set data, aos_matrix<index_type> &&indices, std::shared_ptr<profiler> profiler);
    /**
     * @brief Create a new nearest-neighbor result by providing the calculated indices and distances.
     * @param[in] comm the associated @ref sycl_lsh::mpi::communicator
     * @param[in] data the queried @ref sycl_lsh::data_set
     * @param[in] indices the calculated nearest-neighbor indices
     * @param[in] distances the calculated nearest-neighbor distances
     * @param[in] profiler the optional @ref sycl_lsh::profiler
     */
    nearest_neighbors_result(mpi::communicator comm, data_set data, aos_matrix<index_type> &&indices, aos_matrix<real_type> &&distances, std::shared_ptr<profiler> profiler);

    /// THe associated @ref sycl_lsh::mpi::communicator.
    mpi::communicator comm_;

    /// The @ref sycl_lsh::data_set for which the nearest-neighbors were calculated.
    data_set data_;

    /// Indices of the nearest points in the population @ref sycl_lsh::aos_matrix.
    aos_matrix<index_type> indices_;
    /// The @ref sycl_lsh::aos_matrix representing the lengths to nearest-neighbors points, only present if return_distance was set to true.
    std::optional<aos_matrix<real_type>> distances_;

    /// The optional @ref sycl_lsh::profiler.
    std::shared_ptr<profiler> profiler_{ nullptr };
};

}  // namespace sycl_lsh

#endif  // SYCL_LSH_NEAREST_NEIGHBORS_RESULT_HPP
