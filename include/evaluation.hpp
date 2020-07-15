/**
 * @file
 * @author Marcel Breyer
 * @date 2020-07-15
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
 * @param[in] data_buffer the data set
 * @param[in] communicator the used MPI communicator
 * @return the calculated error ratio
 */
template <typename Knns, typename real_type, typename index_type>
[[nodiscard]] double error_ratio(Knns& knns, mpi_buffers<real_type, index_type>& data_buffer, const MPI_Comm& communicator) {
    static_assert(std::is_base_of_v<detail::knn_base, Knns>, "The first template parameter must by a 'knn' type!");

    const auto& data = knns.get_data();
    const index_type rank_size = data.rank_size;
    const index_type dims = data.dims;
    const index_type k = knns.k;

    std::vector<index_type>& calculated_knns = knns.buffers.active();
    std::vector<double> calculated_knns_dist(calculated_knns.size(), 0.0);
    std::vector<index_type>& correct_knns = knns.buffers.inactive();
    std::vector<double> correct_knns_dist(correct_knns.size(), 0.0);

    int comm_size, comm_rank;
    MPI_Comm_size(communicator, &comm_size);
    MPI_Comm_rank(communicator, &comm_rank);

    for (int round = 0; round < comm_size; ++round) {
        detail::mpi_print(comm_rank, "Round {} of {}\n", round + 1, comm_size);

        const int data_rank = (comm_rank + round) % comm_size;
        int correct_rank_size = data.total_size / comm_size;
        if ((data.total_size % comm_size) != 0 && data_rank < (static_cast<int>(data.total_size) % comm_size)) ++correct_rank_size;
        const index_type data_id_lower = data.total_size / comm_size * data_rank + std::min<index_type>(data_rank, data.total_size % comm_size);
        const index_type data_id_upper = data_id_lower + correct_rank_size;

        for (index_type nn = 0; nn < calculated_knns.size(); ++nn) {
            // check if data point is currently available
            if (calculated_knns[nn] >= data_id_lower && calculated_knns[nn] < data_id_upper) {
                const index_type point = nn / k;
                for (index_type dim = 0; dim < dims; ++dim) {
                    const real_type point_dim = data_buffer.active()[data.get_linear_id(comm_rank, point, rank_size, dim, dims)];
                    const real_type knn_dim = data_buffer.active()[data.get_linear_id(comm_rank, calculated_knns[nn] % rank_size, rank_size, dim, dims)];
                    calculated_knns_dist[nn] += (point_dim - knn_dim) * (point_dim - knn_dim);
                }
                calculated_knns_dist[nn] = std::sqrt(calculated_knns_dist[nn]);
            }

            if (correct_knns[nn] >= data_id_lower && correct_knns[nn] < data_id_upper) {
                const index_type point = nn / k;
                for (index_type dim = 0; dim < dims; ++dim) {
                    const real_type point_dim = data_buffer.active()[data.get_linear_id(comm_rank, point, rank_size, dim, dims)];
                    const real_type knn_dim = data_buffer.active()[data.get_linear_id(comm_rank, correct_knns[nn] % rank_size, rank_size, dim, dims)];
                    correct_knns_dist[nn] += (point_dim - knn_dim) * (point_dim - knn_dim);
                }
                correct_knns_dist[nn] = std::sqrt(correct_knns_dist[nn]);
            }
        }

        // send data to next rank
        data_buffer.send_receive();

        // wait until ALL communication has finished
        MPI_Barrier(communicator);
    }

    index_type mean_error_count = 0;
    real_type mean_error_ratio = 0.0;

    std::vector<real_type> calculated_knns_dist_sorted(k);
    std::vector<real_type> correct_knns_dist_sorted(k);

    for (index_type point = 0; point < rank_size; ++point) {
        for (index_type nn = 0; nn < k; ++nn) {
            calculated_knns_dist_sorted[nn] = calculated_knns_dist[knns.get_linear_id(comm_rank, point, nn, data, k)];
            correct_knns_dist_sorted[nn] = correct_knns_dist[knns.get_linear_id(comm_rank, point, nn, data, k)];
        }
        std::sort(calculated_knns_dist_sorted.begin(), calculated_knns_dist_sorted.end());
        std::sort(correct_knns_dist_sorted.begin(), correct_knns_dist_sorted.end());

        // TODO 2020-06-04 18:01 marcel: penalty
        index_type error_count = 0;
        real_type error_ratio = 0.0;
        for (index_type nn = 0; nn < k; ++nn) {
            if (calculated_knns_dist_sorted[nn] != 0.0 && correct_knns_dist_sorted[nn] != 0.0) {
                ++error_count;
                error_ratio += calculated_knns_dist_sorted[nn] / correct_knns_dist_sorted[nn];
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
