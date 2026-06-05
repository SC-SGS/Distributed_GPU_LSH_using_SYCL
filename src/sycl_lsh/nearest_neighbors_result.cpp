/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/nearest_neighbors_result.hpp"

#include "sycl_lsh/constants.hpp"                           // sycl_lsh::index_type, sycl_lsh::real_type
#include "sycl_lsh/data_set.hpp"                            // sycl_lsh::data_set
#include "sycl_lsh/exceptions/exceptions.hpp"               // sycl_lsh::exception
#include "sycl_lsh/matrix.hpp"                              // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"                    // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/file_parser/file.hpp"         // sycl_lsh::mpi::detail::file
#include "sycl_lsh/mpi/detail/file_parser/file_parser.hpp"  // sycl_lsh::mpi::detail::file_parser
#include "sycl_lsh/mpi/detail/math.hpp"                     // sycl_lsh::mpi::detail::sum
#include "sycl_lsh/mpi/detail/timer.hpp"                    // sycl_lsh::mpi::detail::timer
#include "sycl_lsh/profiler.hpp"                            // sycl_lsh::profiler
#include "sycl_lsh/shape.hpp"                               // sycl_lsh::shape

#include "fmt/format.h"  // fmt::format

#include <algorithm>  // std::transform, std::count, std::sort
#include <cstddef>    // std::size_t
#include <limits>     // std::numeric_limits::max
#include <optional>   // std::nullopt
#include <string>     // std::string
#include <tuple>      // std::tuple, std::make_tuple
#include <utility>    // std::move
#include <vector>     // std::vector

namespace sycl_lsh {

nearest_neighbors_result::nearest_neighbors_result(const mpi::communicator comm, data_set data, aos_matrix<index_type> &&indices, std::shared_ptr<profiler> profiler) :
    comm_{ comm },
    data_{ std::move(data) },
    indices_{ std::move(indices) },
    distances_{ std::nullopt },
    profiler_{ std::move(profiler) } { }

nearest_neighbors_result::nearest_neighbors_result(const mpi::communicator comm, data_set data, aos_matrix<index_type> &&indices, aos_matrix<real_type> &&distances, std::shared_ptr<profiler> profiler) :
    comm_{ comm },
    data_{ std::move(data) },
    indices_{ std::move(indices) },
    distances_{ std::move(distances) },
    profiler_{ std::move(profiler) } { }

std::vector<index_type> nearest_neighbors_result::indices(const std::size_t idx) const {
    if (idx >= indices_.num_rows()) {
        throw exception{ fmt::format("Index out-of-range!: {} >= {}", idx, indices_.num_rows()) };
    }
    std::vector<index_type> temp(indices_.num_cols());
    for (std::size_t nn = 0; nn < temp.size(); ++nn) {
        temp[nn] = indices_(idx, nn);
    }
    return temp;
}

std::vector<real_type> nearest_neighbors_result::distances(const std::size_t idx) const {
    if (!this->has_distances()) {
        throw exception{ "Distances not requested using \"return_distance\". Therefore, they can't be queried!" };
    }
    if (idx >= distances_->num_rows()) {
        throw exception{ fmt::format("Index out-of-range!: {} >= {}", idx, distances_->num_rows()) };
    }
    std::vector<real_type> temp(distances_->num_cols());
    for (std::size_t nn = 0; nn < temp.size(); ++nn) {
        temp[nn] = (*distances_)(idx, nn);
    }
    return temp;
}

void nearest_neighbors_result::save_indices(const std::string &filename, const mpi::file_parser_type file_parser) const {
    const mpi::detail::timer mpi_timer{ comm_ };

    const auto file_writer = mpi::detail::make_file_parser<index_type>(filename, file_parser, mpi::detail::file::mode::write, comm_);
    file_writer->write_content(data_.get_attributes().total_size, indices_.num_cols(), indices_);

    // add entry if available
    if (profiler_ != nullptr) {
        profiler_->add_entry("save_indices", "total_runtime", mpi_timer.elapsed());
        profiler_->add_entry("save_indices", "file_parser", file_parser);
        profiler_->add_entry("save_indices", data_.get_attributes());
    }
}

void nearest_neighbors_result::save_distances(const std::string &filename, const mpi::file_parser_type file_parser) const {
    if (!this->has_distances()) {
        throw exception{ "Distances not requested using \"return_distance\". Therefore, they can't be saved!" };
    }
    const mpi::detail::timer mpi_timer{ comm_ };

    const auto file_writer = mpi::detail::make_file_parser<real_type>(filename, file_parser, mpi::detail::file::mode::write, comm_);
    file_writer->write_content(data_.get_attributes().total_size, distances_->num_cols(), distances_.value());

    // add entry if available
    if (profiler_ != nullptr) {
        profiler_->add_entry("save_distances", "total_runtime", mpi_timer.elapsed());
        profiler_->add_entry("save_distances", "file_parser", file_parser);
        profiler_->add_entry("save_distances", data_.get_attributes());
    }
}

real_type nearest_neighbors_result::recall(const aos_matrix<index_type> &correct_indices) const {
    // perform sanity checks
    if (indices_.shape() != correct_indices.shape()) {
        throw exception{ fmt::format("The index sizes missmatch!: {} != {}", indices_.shape(), correct_indices.shape()) };
    }
    const mpi::detail::timer mpi_timer{ comm_ };

    index_type count = 0;
    for (index_type point = 0; point < indices_.num_rows(); ++point) {
        for (index_type nn = 0; nn < indices_.num_cols(); ++nn) {
            // get calculated nearest-neighbor index
            const index_type calculated_index = indices_(point, nn);
            // check if calculated index is contained in the correct indices
            for (index_type i = 0; i < indices_.num_cols(); ++i) {
                if (calculated_index == correct_indices(point, i)) {
                    // correct ID found
                    ++count;
                    break;
                }
            }
        }
    }

    // gather the results from all MPI ranks
    const std::size_t total_size = mpi::detail::sum(indices_.num_rows(), comm_);
    const real_type res = (static_cast<real_type>(mpi::detail::sum(count, comm_)) / static_cast<real_type>(total_size * indices_.num_cols())) * real_type{ 100.0 };

    // add entry if available
    if (profiler_ != nullptr) {
        profiler_->add_entry("recall", "recall", res);
        profiler_->add_entry("recall", "total_runtime", mpi_timer.elapsed());
        profiler_->add_entry("recall", data_.get_attributes());
    }

    return res;
}

real_type nearest_neighbors_result::recall(const std::string &filename, const mpi::file_parser_type file_parser) const {
    const auto parser = mpi::detail::make_file_parser<index_type>(filename, file_parser, mpi::detail::file::mode::read, comm_);
    const aos_matrix<index_type> correct_indices = parser->parse_content();
    return this->recall(correct_indices);
}

std::tuple<real_type, index_type, index_type> nearest_neighbors_result::error_ratio(const aos_matrix<real_type> &correct_distances) const {
    if (!this->has_distances()) {
        throw exception{ "Distances not requested using \"return_distance\". Therefore, the error_ratio cannot be calculated!" };
    }

    // perform sanity checks
    if (distances_->shape() != correct_distances.shape()) {
        throw exception{ fmt::format("The index sizes missmatch!: {} != {}", distances_->shape(), correct_distances.shape()) };
    }
    const mpi::detail::timer mpi_timer{ comm_ };

    // calculate error ratio
    index_type num_points_not_found = 0;
    index_type num_knn_not_found = 0;
    index_type mean_error_count = 0;
    real_type mean_error_ratio = 0.0;

    std::vector<real_type> calculated_knn_dist_sorted(distances_->num_cols());
    std::vector<real_type> correct_knn_dist_sorted(distances_->num_cols());

    for (index_type point = 0; point < distances_->num_rows(); ++point) {
        // fill k-nearest-neighbor distances for current point
        for (index_type nn = 0; nn < distances_->num_cols(); ++nn) {
            calculated_knn_dist_sorted[nn] = (*distances_)(point, nn);
            correct_knn_dist_sorted[nn] = correct_distances(point, nn);
        }
        // check whether k nearest-neighbor could be found
        if (const auto count_not_found = std::count(calculated_knn_dist_sorted.cbegin(), calculated_knn_dist_sorted.cend(), std::numeric_limits<real_type>::max()); count_not_found != 0) {
            ++num_points_not_found;
            num_knn_not_found += count_not_found;
            continue;
        }
        // sort distances
        std::sort(calculated_knn_dist_sorted.begin(), calculated_knn_dist_sorted.end());
        std::sort(correct_knn_dist_sorted.begin(), correct_knn_dist_sorted.end());

        // calculate error ratio
        index_type error_count = 0;
        real_type error_ratio = 0.0;
        for (index_type nn = 0; nn < distances_->num_cols(); ++nn) {
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
    const real_type avg_mean_error_ratio = mpi::detail::sum(mean_error_ratio, comm_) / static_cast<real_type>(mpi::detail::sum(mean_error_count, comm_));
    const index_type total_num_points_not_found = mpi::detail::sum(num_points_not_found, comm_);
    const index_type total_num_knn_not_found = mpi::detail::sum(num_knn_not_found, comm_);

    // add entry if available
    if (profiler_ != nullptr) {
        profiler_->add_entry("error_ratio", "avg_mean_error_ratio", avg_mean_error_ratio);
        profiler_->add_entry("error_ratio", "total_num_points_not_found", total_num_points_not_found);
        profiler_->add_entry("error_ratio", "total_num_knn_not_found", total_num_knn_not_found);
        profiler_->add_entry("error_ratio", "total_runtime", mpi_timer.elapsed());
        profiler_->add_entry("error_ratio", data_.get_attributes());
    }

    return std::make_tuple(avg_mean_error_ratio, total_num_points_not_found, total_num_knn_not_found);
}

std::tuple<real_type, index_type, index_type> nearest_neighbors_result::error_ratio(const std::string &filename, const mpi::file_parser_type file_parser) const {
    const auto parser = mpi::detail::make_file_parser<real_type>(filename, file_parser, mpi::detail::file::mode::read, comm_);
    const aos_matrix<real_type> correct_distances = parser->parse_content();
    return this->error_ratio(correct_distances);
}

}  // namespace sycl_lsh
