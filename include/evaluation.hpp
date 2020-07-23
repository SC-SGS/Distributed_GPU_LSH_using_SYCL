/**
 * @file
 * @author Marcel Breyer
 * @date 2020-07-23
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
#include <detail/mpi_type.hpp>
#include <knn.hpp>


/**
 * @brief Sums the given @p values over all MPI ranks.
 * @tparam real_type the type of the @p value
 * @param[in] communicator the MPI_Comm communicator
 * @param[in] value the value to sum
 * @return the sum value
 */
template <typename real_type>
[[nodiscard]] real_type sum(const MPI_Comm& communicator, real_type value) {
    real_type sums = 0.0;
    MPI_Allreduce(&value, &sums, 1, detail::mpi_type_cast<real_type>(), MPI_SUM, communicator);
    return sums;
}

/**
 * @brief Averages the given @p values over all MPI ranks.
 * @tparam real_type the type of the @p value
 * @param[in] communicator the MPI_Comm communicator
 * @param[in] value the value to average
 * @return the average value
 */
template <typename real_type>
[[nodiscard]] real_type average(const MPI_Comm& communicator, real_type value) {
    real_type sums = sum(communicator, value);

    int comm_size;
    MPI_Comm_size(communicator, &comm_size);

    return sums / comm_size;
}


/**
 * @brief Calculates the recall using: \f$ \frac{true\ positives}{relevant\ elements} \f$
 * @tparam Knn represents the calculated nearest neighbors
 * @param[in] knns the calculated and correct k-nearest-neighbors
 * @param[in] comm_rank the current MPI rank
 * @return the calculated recall
 */
template <typename Knns>
[[nodiscard]] typename Knns::real_type recall(Knns& knns, const int comm_rank) {
    static_assert(std::is_base_of_v<detail::knn_base, Knns>, "The first template parameter must by a 'knn' type!");

    using index_type = typename Knns::index_type;
    using real_type = typename Knns::real_type;

    const auto& data = knns.get_data();
    const index_type size = data.rank_size;
    const index_type k = knns.k;
    real_type average_recall = 0.0;

    std::vector<index_type>& calculated_knns = knns.buffers_knn.active();
    std::vector<index_type>& correct_knns = knns.buffers_knn.inactive();
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
 * @tparam real_type a floating point type
 * @tparam index_type an integer type
 * @param[in] knns the calculated and correct k-nearest-neighbors
 * @param[in] data_buffer the data set
 * @param[in] comm_rank the current MPI rank
 * @return the calculated error ratio
 */
template <typename Knns, typename real_type, typename index_type>
[[nodiscard]] std::pair<real_type, index_type> error_ratio(Knns& knns, mpi_buffers<real_type, index_type>& data_buffer, const int comm_rank) {
    static_assert(std::is_base_of_v<detail::knn_base, Knns>, "The first template parameter must by a 'knn' type!");

    const auto& data = knns.get_data();
    const index_type rank_size = data.rank_size;
    const index_type k = knns.k;

    std::vector<real_type>& calculated_knns_dist = knns.buffers_dist.active();
    std::vector<real_type>& correct_knns_dist = knns.buffers_dist.inactive();

    index_type num_not_found = 0;
    index_type mean_error_count = 0;
    real_type mean_error_ratio = 0.0;

    std::vector<real_type> calculated_knns_dist_sorted(k);
    std::vector<real_type> correct_knns_dist_sorted(k);

    for (index_type point = 0; point < rank_size; ++point) {
        for (index_type nn = 0; nn < k; ++nn) {
            calculated_knns_dist_sorted[nn] = calculated_knns_dist[knns.get_linear_id(comm_rank, point, nn, data, k)];
            if (calculated_knns_dist_sorted[nn] == std::numeric_limits<real_type>::max()) {
                calculated_knns_dist_sorted[nn] = 0.0;
                ++num_not_found;
            } else {
                calculated_knns_dist_sorted[nn] = std::sqrt(calculated_knns_dist_sorted[nn]);
            }
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

    // TODO 2020-07-16 17:12 marcel: what to return?
    real_type error_ratio_percent = std::abs((mean_error_ratio / mean_error_count) * 100 - 100);
    return std::make_pair(error_ratio_percent, num_not_found);
}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_EVALUATION_HPP
