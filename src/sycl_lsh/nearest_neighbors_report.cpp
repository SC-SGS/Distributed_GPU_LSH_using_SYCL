/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/nearest_neighbors_report.hpp"

#include "sycl_lsh/constants.hpp"           // sycl_lsh::index_type, sycl_lsh::real_type
#include "sycl_lsh/detail/shape.hpp"        // sycl_lsh::detail::shape
#include "sycl_lsh/matrix.hpp"              // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"    // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/logging.hpp"  // sycl_lsh::mpi::detail::log
#include "sycl_lsh/mpi/detail/math.hpp"     // sycl_lsh::mpi::detail::sum
#include "sycl_lsh/mpi/detail/timer.hpp"    // sycl_lsh::mpi::detail::timer

#include "fmt/format.h"  // fmt::format

#include <algorithm>  // std::copy, std::transform, std::count, std::sort
#include <cmath>      // std::sqrt
#include <limits>     // std::numeric_limits::max
#include <string>     // std::string
#include <tuple>      // std::tuple, std::make_tuple
#include <vector>     // std::vector

namespace sycl_lsh::report {

real_type recall(const aos_matrix<index_type> &calculated_indices, const aos_matrix<index_type> &correct_indices, const mpi::communicator &comm) {
    const mpi::detail::timer mpi_timer{ comm };

    // perform sanity checks
    if (calculated_indices.shape() != correct_indices.shape()) {
        throw exception{ fmt::format("The index sizes missmatch!: {} != {}", calculated_indices.shape(), correct_indices.shape()) };
    }

    index_type count = 0;
    for (index_type point = 0; point < calculated_indices.num_rows(); ++point) {
        for (index_type nn = 0; nn < calculated_indices.num_cols(); ++nn) {
            // get calculated nearest-neighbor index
            const index_type calculated_index = calculated_indices(point, nn);
            // check if calculated index is contained in the correct indices
            for (index_type i = 0; i < calculated_indices.num_cols(); ++i) {
                if (calculated_index == correct_indices(point, i)) {
                    // correct ID found
                    ++count;
                    break;
                }
            }
        }
    }

    // gather the results from all MPI ranks
    const std::size_t total_size = mpi::detail::sum(calculated_indices.num_rows(), comm);
    const real_type res = (static_cast<real_type>(mpi::detail::sum(count, comm)) / (total_size * calculated_indices.num_cols())) * real_type{ 100.0 };

    mpi::detail::log(comm, "\nCalculated recall in {}.\n", mpi_timer.elapsed());
    return res;
}

std::tuple<real_type, index_type, index_type> error_ratio(const aos_matrix<real_type> &calculated_distances, const aos_matrix<real_type> &correct_distances, const mpi::communicator &comm) {
    const mpi::detail::timer mpi_timer{ comm };

    // perform sanity checks
    if (calculated_distances.shape() != correct_distances.shape()) {
        throw exception{ fmt::format("The index sizes missmatch!: {} != {}", calculated_distances.shape(), correct_distances.shape()) };
    }

    // calculate error ratio
    index_type num_points_not_found = 0;
    index_type num_knn_not_found = 0;
    index_type mean_error_count = 0;
    real_type mean_error_ratio = 0.0;

    std::vector<real_type> calculated_knn_dist_sorted(calculated_distances.num_cols());
    std::vector<real_type> correct_knn_dist_sorted(calculated_distances.num_cols());

    for (index_type point = 0; point < calculated_distances.num_rows(); ++point) {
        // fill k-nearest-neighbor distances for current point
        for (index_type nn = 0; nn < calculated_distances.num_cols(); ++nn) {
            calculated_knn_dist_sorted[nn] = calculated_distances(point, nn);
            correct_knn_dist_sorted[nn] = correct_distances(point, nn);
        }
        // check whether k nearest-neighbor could be found
        if (const auto count_not_found = std::count(calculated_knn_dist_sorted.cbegin(), calculated_knn_dist_sorted.cend(), std::numeric_limits<real_type>::max()); count_not_found != 0) {
            ++num_points_not_found;
            num_knn_not_found += count_not_found;
            continue;
        }
        // calculate `std::sqrt` distance
        std::transform(calculated_knn_dist_sorted.begin(), calculated_knn_dist_sorted.end(), calculated_knn_dist_sorted.begin(), [](const real_type val) { return std::sqrt(val); });
        // sort distances
        std::sort(calculated_knn_dist_sorted.begin(), calculated_knn_dist_sorted.end());
        std::sort(correct_knn_dist_sorted.begin(), correct_knn_dist_sorted.end());

        // calculate error ratio
        index_type error_count = 0;
        real_type error_ratio = 0.0;
        for (index_type nn = 0; nn < calculated_distances.num_cols(); ++nn) {
            if (correct_knn_dist_sorted[nn] == real_type{ 0.0 }) {
                // two different points at the same position
                if (calculated_knn_dist_sorted[nn] == real_type{ 0.0 }) {
                    // calculated nearest neighbor is correct
                    error_ratio += real_type{ 1.0 };
                    ++error_count;
                }
            } else {
                // calculate distance ratio
                error_ratio += calculated_knn_dist_sorted[nn] / correct_knn_dist_sorted[nn];
                ++error_count;
            }
        }
        // calculate error ratio for current k-nearest neighbors
        if (error_count != index_type{ 0 }) {
            mean_error_ratio += error_ratio / static_cast<real_type>(error_count);
            ++mean_error_count;
        }
    }

    // collect results from each MPI rank
    const real_type avg_mean_error_ratio = mpi::detail::sum(mean_error_ratio, comm) / mpi::detail::sum(mean_error_count, comm);
    const index_type total_num_points_not_found = mpi::detail::sum(num_points_not_found, comm);
    const index_type total_num_knn_not_found = mpi::detail::sum(num_knn_not_found, comm);

    mpi::detail::log(comm, "\nCalculated error ration in {}.\n", mpi_timer.elapsed());
    return std::make_tuple(avg_mean_error_ratio, total_num_points_not_found, total_num_knn_not_found);
}

}  // namespace sycl_lsh::report
