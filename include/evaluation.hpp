/**
 * @file
 * @author Marcel Breyer
 * @date 2020-08-28
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
#include <tuple>
#include <vector>

#include <config.hpp>
#include <data.hpp>
#include <detail/mpi_type.hpp>
#include <knn.hpp>


/**
 * @brief Sums the given @p values over all MPI ranks.
 * @tparam T the type of the @p value
 * @param[in] value the value to sum
 * @param[in] communicator the MPI_Comm communicator
 * @return the sum value
 */
template <typename T>
[[nodiscard]] T mpi_sum(T value, const MPI_Comm& communicator) {
    T sums = 0.0;
    MPI_Allreduce(&value, &sums, 1, detail::mpi_type_cast<T>(), MPI_SUM, communicator);
    return sums;
}

/**
 * @brief Averages the given @p values over all MPI ranks.
 * @tparam T the type of the @p value
 * @param[in] value the value to average
 * @param[in] communicator the MPI_Comm communicator
 * @return the average value
 */
template <typename T>
[[nodiscard]] T average(T value, const MPI_Comm& communicator) {
    T sums = mpi_sum(value, communicator);

    int comm_size;
    MPI_Comm_size(communicator, &comm_size);

    return sums / comm_size;
}


/**
 * @brief Calculates the recall using: \f$ \frac{true\ positives}{relevant\ elements} \f$
 * @tparam Knn represents the calculated nearest neighbors
 * @param[in] knns the calculated and correct k-nearest-neighbors
 * @param[in] comm_rank the current MPI rank
 * @param[in] comm_size the current MPI_Comm size
 * @param[in] communicator the used MPI_Comm communicator
 * @return the calculated recall
 */
template <typename Knns>
[[nodiscard]] typename Knns::real_type recall(Knns& knns, const int comm_rank, const int comm_size, const MPI_Comm& communicator) {
    static_assert(std::is_base_of_v<detail::knn_base, Knns>, "The first template parameter must by a 'knn' type!");

    using index_type = typename Knns::index_type;
    using real_type = typename Knns::real_type;
    using aos_layout = knn<memory_layout::aos, typename Knns::options_type, typename Knns::data_type>;
    
    const auto& data = knns.get_data();
    const bool has_smaller_rank_size = (data.total_size % comm_size != 0) && static_cast<index_type>(comm_rank) >= (data.total_size % comm_size);
    const index_type size = has_smaller_rank_size ? data.rank_size - 1 : data.rank_size;
    const index_type k = knns.k;
    index_type count = 0;

    std::vector<index_type>& calculated_knns = knns.buffers_knn.active();
    std::vector<index_type>& correct_knns = knns.buffers_knn.inactive();

    for (index_type point = 0; point < size; ++point) {
        for (index_type i = 0; i < k; ++i) {
            const index_type calculated_id = calculated_knns[knns.get_linear_id(comm_rank, point, i, data, k)];
            for (index_type j = 0; j < k; ++j) {
                if (calculated_id == correct_knns[aos_layout::get_linear_id(comm_rank, point, j, data, k)]) {
                    ++count;
                    break;
                }
            }
        }
    }

    return (static_cast<real_type>(mpi_sum(count, communicator)) / (data.total_size * k)) * 100;
}

/**
 * @brief Calculates the error ratio using: \f$ \frac{1}{N} \cdot \sum\limits_{i = 0}^N (\frac{1}{k} \cdot \sum\limits_{j = 0}^k \frac{dist_{LSH_j}}{dist_{correct_j}}) \f$
 * @tparam Knn represents the calculated nearest neighbors
 * @tparam real_type a floating point type
 * @tparam index_type an integer type
 * @param[in] knns the calculated and correct k-nearest-neighbors
 * @param[in] data_buffer the data set
 * @param[in] comm_rank the current MPI rank
 * @param[in] communicator the current MPI_Comm communicator
 * @return the calculated error ratio, the number of points for which only less than `k` nearest-neighbors could be found and the total
 *         number of nearest-neighbors that couldn't be found
 */
template <typename Knns, typename real_type, typename index_type>
[[nodiscard]] std::tuple<real_type, index_type, index_type> error_ratio(Knns& knns, mpi_buffers<real_type, index_type>& data_buffer, const int comm_rank, const MPI_Comm& communicator) {
    static_assert(std::is_base_of_v<detail::knn_base, Knns>, "The first template parameter must by a 'knn' type!");

    using aos_layout = knn<memory_layout::aos, typename Knns::options_type, typename Knns::data_type>;

    const auto& data = knns.get_data();
    const index_type rank_size = data.rank_size;
    const index_type k = knns.k;

    std::vector<real_type>& calculated_knns_dist = knns.buffers_dist.active();
    std::vector<real_type>& correct_knns_dist = knns.buffers_dist.inactive();

    index_type num_points_not_found = 0;
    index_type num_knn_not_found = 0;
    index_type mean_error_count = 0;
    real_type mean_error_ratio = 0.0;

    std::vector<real_type> calculated_knns_dist_sorted(k);
    std::vector<real_type> correct_knns_dist_sorted(k);

    for (index_type point = 0; point < rank_size; ++point) {
        for (index_type nn = 0; nn < k; ++nn) {
            calculated_knns_dist_sorted[nn] = calculated_knns_dist[knns.get_linear_id(comm_rank, point, nn, data, k)];
            correct_knns_dist_sorted[nn] = correct_knns_dist[aos_layout::get_linear_id(comm_rank, point, nn, data, k)];
        }
        auto count = std::count(calculated_knns_dist_sorted.cbegin(), calculated_knns_dist_sorted.cend(), std::numeric_limits<real_type>::max());
        if (count != 0) {
            ++num_points_not_found;
            num_knn_not_found += count;
            continue;
        }
        std::transform(calculated_knns_dist_sorted.begin(), calculated_knns_dist_sorted.end(), calculated_knns_dist_sorted.begin(),
                   [](const real_type val) { return std::sqrt(val); });
        std::sort(calculated_knns_dist_sorted.begin(), calculated_knns_dist_sorted.end());
        std::sort(correct_knns_dist_sorted.begin(), correct_knns_dist_sorted.end());

        index_type error_count = 0;
        real_type error_ratio = 0.0;
        for (index_type nn = 0; nn < k; ++nn) {
            if (calculated_knns_dist_sorted[nn] != 0.0 && correct_knns_dist_sorted[nn] != 0.0) {
                ++error_count;
                error_ratio += calculated_knns_dist_sorted[nn] / correct_knns_dist_sorted[nn];
            } else {
                ++error_count;
                ++error_ratio;
            }
        }
        if (error_count != 0) {
            ++mean_error_count;
            mean_error_ratio += error_ratio / error_count;
        }
    }

    const real_type avg_mean_error_ratio = average(mean_error_ratio / mean_error_count, communicator);
    const index_type total_num_points_not_found = mpi_sum(num_points_not_found, communicator);
    const index_type total_num_knn_not_found = mpi_sum(num_knn_not_found, communicator);

    // TODO 2020-08-28 14:46 marcel: check again
    return std::make_tuple(avg_mean_error_ratio, total_num_points_not_found, total_num_knn_not_found);
}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_EVALUATION_HPP
