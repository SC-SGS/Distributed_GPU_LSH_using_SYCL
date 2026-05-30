/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/knn.hpp"

#include "sycl_lsh/data_attributes.hpp"       // sycl_lsh::data_attributes
#include "sycl_lsh/data_set.hpp"              // sycl_lsh::data_set
#include "sycl_lsh/detail/shape.hpp"          // sycl_lsh::detail::shape
#include "sycl_lsh/mpi/communicator.hpp"      // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/math.hpp"       // sycl_lsh::mpi::detail::sum
#include "sycl_lsh/mpi/detail/type_cast.hpp"  // sycl_lsh::mpi::detail::mpi_datatype
#include "sycl_lsh/mpi/file_parser/file.hpp"  // sycl_lsh::mpi::file::mode
#include "sycl_lsh/mpi/logger.hpp"            // sycl_lsh::mpi::logger
#include "sycl_lsh/mpi/timer.hpp"             // sycl_lsh::mpi::timer
#include "sycl_lsh/options.hpp"               // sycl_lsh::options

#include "fmt/format.h"  // fmt::format
#include "mpi.h"         // MPI_Sendrecv_replace, MPI_STATUS_IGNORE

#include "../../include/sycl_lsh/matrix.hpp"
#include <algorithm>  // std::copy, std::transform, std::count, std::sort
#include <cmath>      // std::sqrt
#include <limits>     // std::numeric_limits::max
#include <string>     // std::string
#include <tuple>      // std::tuple, std::make_tuple
#include <vector>     // std::vector

namespace sycl_lsh {

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //

knn::knn(const options &opt, const data_set &data, const mpi::communicator &comm, const mpi::logger &logger) :
    knn{ opt.k, data, comm, logger } { }

knn::knn(const index_type k, const data_set &data, const mpi::communicator &comm, const mpi::logger &logger) :
    attr_{ data.get_attributes() },
    comm_{ comm },
    logger_{ logger },
    k_{ k },
    knn_indices_{ detail::shape{ attr_.rank_size, k } },
    knn_distances_{ detail::shape{ attr_.rank_size, k }, std::numeric_limits<real_type>::max() } {
    const mpi::timer mpi_timer{ comm_ };

    SYCL_LSH_ASSERT(0 < k, "Illegal number of k-nearest-neighbors!");

    // calculate start ID
    const index_type base_id = comm_.rank() * attr_.rank_size;

    // fill default values
    for (index_type point = 0; point < attr_.rank_size; ++point) {
        for (index_type nn = 0; nn < k_; ++nn) {
            knn_indices_(point, nn) = base_id + point;
        }
    }

    // correctly set default values for dummy points on last MPI rank
    if (comm_.rank() == comm_.size() - 1) {
        const index_type correct_rank_size = attr_.total_size - ((comm_.size() - 1) * attr_.rank_size);
        for (index_type point = correct_rank_size; point < attr_.rank_size; ++point) {
            for (index_type nn = 0; nn < k_; ++nn) {
                knn_indices_(point, nn) = base_id + correct_rank_size - 1;
            }
        }
    }

    logger_.log("Created knn object in {}.\n", mpi_timer.elapsed());
}

// ---------------------------------------------------------------------------------------------------------- //
//                                                knn results                                                 //
// ---------------------------------------------------------------------------------------------------------- //
[[nodiscard]] std::vector<index_type> knn::get_knn_ids(const index_type point) const {
    SYCL_LSH_ASSERT(0 <= point && point < attr_.rank_size, "Out-of-bounce access for data point!");

    std::vector<index_type> res(k_);
    for (index_type nn = 0; nn < k_; ++nn) {
        res[nn] = knn_indices_(point, nn);
    }
    return res;
}

[[nodiscard]] std::vector<real_type> knn::get_knn_dists(const index_type point) const {
    SYCL_LSH_ASSERT(0 <= point && point < attr_.rank_size, "Out-of-bounce access for data point!\n");

    std::vector<real_type> res(k_);
    for (index_type nn = 0; nn < k_; ++nn) {
        res[nn] = knn_distances_(point, nn);
    }
    return res;
}

// ---------------------------------------------------------------------------------------------------------- //
//                                                  save knn                                                  //
// ---------------------------------------------------------------------------------------------------------- //
void knn::save_knns(const options &opt) {
    const mpi::timer mpi_timer{ comm_ };

    // check if the required command line argument is present
    if (!opt.knn_save_file.has_value()) {
        throw exception{ "Required command line argument 'knn_save_file' not provided!" };
    }

    // write content to the respective file
    const auto file_parser = mpi::make_file_parser<index_type>(opt.knn_save_file.value(), opt.file_parser, mpi::file::mode::write, comm_, logger_);
    file_parser->write_content(attr_.total_size, k_, knn_indices_);

    logger_.log("Saved k-nearest-neighbor IDs in {}.\n", mpi_timer.elapsed());
}

void knn::save_distances(const options &opt) {
    const mpi::timer mpi_timer{ comm_ };

    // check if the required command line argument is present
    if (!opt.knn_dist_save_file.has_value()) {
        throw exception{ "Required command line argument 'knn_dist_save_file' not provided!" };
    }

    // create a temporary matrix in order to be able to apply the sqrt function
    aos_matrix<real_type> temp_matrix{ knn_distances_ };

    // transform the values using `std::sqrt`
    std::transform(temp_matrix.data(), temp_matrix.data() + temp_matrix.size(), temp_matrix.data(), [](const real_type val) { return std::sqrt(val); });

    // write content to the respective file
    const auto file_parser = mpi::make_file_parser<real_type>(opt.knn_dist_save_file.value(), opt.file_parser, mpi::file::mode::write, comm_, logger_);
    file_parser->write_content(attr_.total_size, k_, temp_matrix);

    logger_.log("Saved k-nearest-neighbor distances in {}.\n", mpi_timer.elapsed());
}

// ---------------------------------------------------------------------------------------------------------- //
//                                                evaluate knn                                                //
// ---------------------------------------------------------------------------------------------------------- //
[[nodiscard]] real_type knn::recall(const options &opt) {
    const mpi::timer mpi_timer{ comm_ };

    // load correct k-nearest-neighbor IDs
    // check if the required command line argument is present
    if (!opt.evaluate_knn_file.has_value()) {
        throw exception{ "Required command line argument 'evaluate_knn_file' not provided!" };
    }

    // read correct k-nearest-neighbor IDs from the respective file
    const std::string &file_name = opt.evaluate_knn_file.value();
    const auto file_parser = mpi::make_file_parser<index_type>(file_name, opt.file_parser, mpi::file::mode::read, comm_, logger_);
    const index_type parsed_total_size = file_parser->parse_total_size();
    const index_type parsed_rank_size = file_parser->parse_rank_size();
    const index_type parsed_dims = file_parser->parse_dims();
    aos_matrix<index_type> correct_knn_indices = file_parser->parse_content();

    // perform sanity checks
    if (parsed_total_size != attr_.total_size) {
        throw exception{ fmt::format("The total number of points in '{}' is {}, but should be {}!", file_name, parsed_total_size, attr_.total_size) };
    }
    if (parsed_rank_size != attr_.rank_size) {
        throw exception{ fmt::format("The number of points per MPI rank in '{}' is {}, but should be {}!", file_name, parsed_rank_size, attr_.rank_size) };
    }
    if (parsed_dims != k_) {
        throw exception{ fmt::format("The number of nearest-neighbors in '{}' is {}, but should be {}!", file_name, parsed_dims, k_) };
    }

    const index_type correct_rank_size = comm_.rank() == comm_.size() - 1 ? (attr_.total_size - (comm_.size() - 1) * attr_.rank_size) : attr_.rank_size;

    index_type count = 0;
    for (index_type point = 0; point < correct_rank_size; ++point) {
        for (index_type nn = 0; nn < k_; ++nn) {
            // get calculated k-nearest-neighbor ID
            const index_type calculated_id = knn_indices_(point, nn);
            // check if calculated ID is contained in the correct IDs
            for (index_type i = 0; i < k_; ++i) {
                if (calculated_id == correct_knn_indices(point, i)) {
                    // correct ID found
                    ++count;
                    break;
                }
            }
        }
    }

    const real_type res = (static_cast<real_type>(mpi::detail::sum(count, comm_)) / (attr_.total_size * k_)) * real_type{ 100.0 };

    logger_.log("\nCalculated recall in {}.\n", mpi_timer.elapsed());
#if defined(SYCL_LSH_BENCHMARK)
    if (comm_.is_main_rank()) {
        mpi::timer::benchmark_out() << res << ',';
    }
#endif

    return res;
}

[[nodiscard]] std::tuple<real_type, index_type, index_type> knn::error_ratio(const options &opt) {
    const mpi::timer mpi_timer{ comm_ };

    // load correct k-nearest-neighbor distances
    // check if the required command line argument is present
    if (!opt.evaluate_knn_dist_file.has_value()) {
        throw exception{ "Required command line argument 'evaluate_knn_dist_file' not provided!" };
    }

    // read correct k-nearest-neighbor distances from the respective file
    const std::string &file_name = opt.evaluate_knn_dist_file.value();
    const auto file_parser = mpi::make_file_parser<real_type>(file_name, opt.file_parser, mpi::file::mode::read, comm_, logger_);
    const index_type parsed_total_size = file_parser->parse_total_size();
    const index_type parsed_rank_size = file_parser->parse_rank_size();
    const index_type parsed_dims = file_parser->parse_dims();
    aos_matrix<real_type> correct_knn_distances = file_parser->parse_content();

    // perform sanity checks
    if (parsed_total_size != attr_.total_size) {
        throw exception{ fmt::format("The total number of points in '{}' is {}, but should be {}!", file_name, parsed_total_size, attr_.total_size) };
    }
    if (parsed_rank_size != attr_.rank_size) {
        throw exception{ fmt::format("The number of points per MPI rank in '{}' is {}, but should be {}!", file_name, parsed_rank_size, attr_.rank_size) };
    }
    if (parsed_dims != k_) {
        throw exception{ fmt::format("The number of nearest-neighbor distances in '{}' is {}, but should be {}!", file_name, parsed_dims, k_) };
    }

    const index_type correct_rank_size = comm_.rank() == comm_.size() - 1 ? (attr_.total_size - (comm_.size() - 1) * attr_.rank_size) : attr_.rank_size;

    // calculate error ratio
    index_type num_points_not_found = 0;
    index_type num_knn_not_found = 0;
    index_type mean_error_count = 0;
    real_type mean_error_ratio = 0.0;

    std::vector<real_type> calculated_knn_dist_sorted(k_);
    std::vector<real_type> correct_knn_dist_sorted(k_);

    for (index_type point = 0; point < correct_rank_size; ++point) {
        // fill k-nearest-neighbor distances for current point
        for (index_type nn = 0; nn < k_; ++nn) {
            calculated_knn_dist_sorted[nn] = knn_distances_(point, nn);
            correct_knn_dist_sorted[nn] = correct_knn_distances(point, nn);
        }
        // check whether k k-nearest-neighbor could be found
        const auto count_not_found = std::count(calculated_knn_dist_sorted.cbegin(), calculated_knn_dist_sorted.cend(), std::numeric_limits<real_type>::max());
        if (count_not_found != 0) {
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
        for (index_type nn = 0; nn < k_; ++nn) {
            if (correct_knn_dist_sorted[nn] == real_type{ 0.0 }) {
                // two different points at the same position
                if (calculated_knn_dist_sorted[nn] == real_type{ 0.0 }) {
                    // calculated nearest neighbor is correct
                    error_ratio += 1.0;
                    ++error_count;
                }
            } else {
                // calculate distance ratio
                error_ratio += calculated_knn_dist_sorted[nn] / correct_knn_dist_sorted[nn];
                ++error_count;
            }
        }
        // calculate error ratio for current k-nearest neighbors
        if (error_count != 0) {
            mean_error_ratio += error_ratio / static_cast<real_type>(error_count);
            ++mean_error_count;
        }
    }

    // collect results from each MPI rank
    const real_type avg_mean_error_ratio = mpi::detail::sum(mean_error_ratio, comm_) / mpi::detail::sum(mean_error_count, comm_);
    const index_type total_num_points_not_found = mpi::detail::sum(num_points_not_found, comm_);
    const index_type total_num_knn_not_found = mpi::detail::sum(num_knn_not_found, comm_);

    logger_.log("\nCalculated error ration in {}.\n", mpi_timer.elapsed());
#if defined(SYCL_LSH_BENCHMARK)
    if (comm_.is_main_rank()) {
        mpi::timer::benchmark_out() << avg_mean_error_ratio << ','
                                    << total_num_points_not_found << ','
                                    << total_num_knn_not_found << ',';
    }
#endif

    return std::make_tuple(avg_mean_error_ratio, total_num_points_not_found, total_num_knn_not_found);
}

}  // namespace sycl_lsh
