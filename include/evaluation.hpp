/**
 * @file
 * @author Marcel Breyer
 * @date 2020-07-09
 *
 * @brief Implements metrics to evaluate the @ref knn search results.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_EVALUATION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_EVALUATION_HPP

#include <algorithm>
#include <cmath>
#include <istream>
#include <iostream>
#include <string>
#include <vector>

#include <config.hpp>
#include <data.hpp>
#include <knn.hpp>


/**
 * @brief Averages the given @p values over all MPI ranks.
 * @param communicator the MPI_Comm communicator
 * @param value the value to average
 * @return the average value
 */
[[nodiscard]] double average(const MPI_Comm& communicator, double value) {
    double sum = 0.0;
    MPI_Reduce(&value, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, communicator);

    int comm_size;
    MPI_Comm_size(communicator, &comm_size);
    double avg = sum / comm_size;
    MPI_Bcast(&avg, 1, MPI_DOUBLE, 0, communicator);

    return avg;
}


/**
 * @brief Calculates the recall using: \f$ \frac{true\ positives}{relevant\ elements} \f$
 * @tparam Knn represents the calculated nearest neighbors
 * @param[in] knns the calculated and correct k-nearest-neighbors
 * @param[in] comm_rank the current MPI rank
 * @return the calculated recall
 */
template <typename Knns>
[[nodiscard]] double recall(Knns& knns, const int comm_rank) {
    static_assert(std::is_base_of_v<detail::knn_base, Knns>, "The first template parameter must by a 'knn' type!");

    using index_type = typename Knns::index_type;
    using real_type = typename Knns::real_type;

    const auto& data = knns.get_data();
    const index_type size = data.rank_size;
    const index_type k = knns.k;
    real_type average_recall = 0.0;

    std::vector<index_type>& calculated_knns = knns.buffers.active();
    std::vector<index_type>& correct_knns = knns.buffers.inactive();
    for (index_type point = 0; point < size; ++point) {
        index_type count = 0;
        for (index_type i = 0; i < k; ++i) {
            const index_type calculated_id = calculated_knns[knns.get_linear_id(comm_rank, point, i, data, k)];
            for (index_type j = 0; j < k; ++j) {
                if (calculated_id == correct_knns[knns.get_linear_id(comm_rank, point, j, data, k)]) {
                    ++count;
                    break;
                }
            }
        }
        average_recall += count / static_cast<real_type>(k);
    }
    return (average_recall / size)  * 100;
}


/**
 * @brief Calculates the error ratio using: \f$ \frac{1}{N} \cdot \sum\limits_{i = 0}^N (\frac{1}{k} \cdot \sum\limits_{j = 0}^k \frac{dist_{LSH_j}}{dist_{correct_j}}) \f$
 * @tparam Knn represents the calculated nearest neighbors
 * @param[in] knns the calculated and correct k-nearest-neighbors
 * @param[in] data the data set
 * @param[in] comm_rank the current MPI rank
 * @return the calculated error ratio
 */
template <typename Knns, typename real_type, typename index_type>
[[nodiscard]] double error_ratio(Knns& knns, mpi_buffers<real_type, index_type>& data, const int comm_rank) {
    static_assert(std::is_base_of_v<detail::knn_base, Knns>, "The first template parameter must by a 'knn' type!");

    const auto& data_ = knns.get_data();
    const std::size_t size = knns.get_data().rank_size;
    const std::size_t dims = knns.get_data().dims;

    std::vector<index_type>& calculated_knns = knns.buffers.active();
    std::vector<index_type>& correct_knns = knns.buffers.inactive();

    std::vector<real_type> dist(knns.k, 0.0);
    std::vector<real_type> ideal_dist(knns.k, 0.0);

    index_type mean_error_count = 0;
    real_type mean_error_ratio = 0.0;


    const auto distances_sorted = [&](const index_type point, const std::vector<index_type>& vec, std::vector<real_type>& dist_vec) {
        std::fill(dist_vec.begin(), dist_vec.end(), 0.0);
        for (index_type nn = 0; nn < knns.k; ++nn) {
            for (index_type dim = 0; dim < dims; ++dim) {
                const real_type point_dim = data.active()[data_.get_linear_id(comm_rank, point, size, dim, dims)];
                const real_type knn_dim = data.active()[data_.get_linear_id(comm_rank, vec[knns.get_linear_id(comm_rank, nn, dim, data_, knns.k)], size, dim, dims)];
                dist_vec[nn] += (point_dim - knn_dim) * (point_dim - knn_dim);
            }
            dist_vec[nn] = std::sqrt(dist[nn]);
        }
        std::sort(dist_vec.begin(), dist_vec.end(), std::greater<>());
    };


    for (index_type point = 0; point < data_.rank_size; ++point) {
        distances_sorted(point, calculated_knns, dist);
        distances_sorted(point, correct_knns, ideal_dist);

        // TODO 2020-06-04 18:01 marcel: penalty
        index_type error_count = 0;
        real_type error_ratio = 0.0;
        for (index_type nn = 0; nn < knns.k; ++nn) {
            if (dist[nn] != 0.0 && ideal_dist[nn] != 0.0) {
                ++error_count;
                error_ratio += dist[nn] / ideal_dist[nn];
            }
        }
        if (error_count != 0) {
            ++mean_error_count;
            mean_error_ratio += error_ratio / error_count;
        }
    }

    return mean_error_ratio / mean_error_count;
}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_EVALUATION_HPP
