/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/nearest_neighbors.hpp"

#include "sycl_lsh/data_set.hpp"                    // sycl_lsh::data_set
#include "sycl_lsh/detail/hashing/hash_tables.hpp"  // sycl_lsh::detail::hashing::hash_tables
#include "sycl_lsh/detail/shape.hpp"                // sycl_lsh::detail::shape
#include "sycl_lsh/matrix.hpp"                      // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"            // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/logging.hpp"          // sycl_lsh::mpi::detail::log
#include "sycl_lsh/mpi/detail/timer.hpp"            // sycl_lsh::mpi::detail::timer
#include "sycl_lsh/nearest_neighbors_result.hpp"    // sycl_lsh::nearest_neighbors_result
#include "sycl_lsh/options.hpp"                     // sycl_lsh::locality_sensitive_hashing_options
#include "sycl_lsh/profiler.hpp"                    // sycl_lsh::profiler

#include "fmt/format.h"  // fmt::format

#include <optional>  // std::nullopt
#include <utility>   // std::move

namespace sycl_lsh {

void nearest_neighbors::fit(data_set X) {
    const mpi::detail::timer mpi_timer{ comm_ };

    // perform some sanity checks
    if (X.get_attributes().rank_size == 0) {
        throw exception{ "The provided training data set is empty!" };
    }

    // store the provided data set internally (so it can be used in kneighbors if no data set has been provided)
    data_ = std::move(X);

    // create the hash tables
    using namespace detail::hashing;
    switch (lsh_options_.hash_function) {
        case hash_function_type::random_projections:
            hash_tables_ = std::make_unique<hash_tables<random_projections>>(lsh_options_, data_, queue_, comm_, profiler_);
            break;
        case hash_function_type::entropy_based:
            hash_tables_ = std::make_unique<hash_tables<entropy_based>>(lsh_options_, data_, queue_, comm_, profiler_);
            break;
        case hash_function_type::mixed_hash_functions:
            hash_tables_ = std::make_unique<hash_tables<mixed_hash_functions>>(lsh_options_, data_, queue_, comm_, profiler_);
            break;
    }

    const auto runtime = mpi_timer.elapsed();
    mpi::detail::log(comm_, "Fit the nearest-neighbors estimator in {}.\n\n", runtime);

    // add entry if available
    if (profiler_ != nullptr) {
        profiler_->add_entry("fit", "total_runtime", runtime);
        profiler_->add_entry("fit", data_.get_attributes());
    }
}

nearest_neighbors_result nearest_neighbors::kneighbors_impl(data_set X, const index_type used_n_neighbors, const bool return_distances) const {
    const mpi::detail::timer mpi_timer{ comm_ };

    const data_set::attributes input_attr = X.get_attributes();

    // perform some sanity checks
    if (hash_tables_ == nullptr) {
        throw exception{ "Hash tables not initialized yet. Did you call fit()?" };
    }
    if (input_attr.rank_size == 0) {
        throw exception{ "The provided training data set is empty!" };
    }
    if (input_attr.dims != data_.get_attributes().dims) {
        throw exception{ fmt::format("The number of dimensions in the training data set ({}) and the provided data set ({}) must be the same!", input_attr.dims, data_.get_attributes().dims) };
    }
    if (used_n_neighbors < 1 || used_n_neighbors > input_attr.rank_size) {
        throw exception{ fmt::format("k ({}) must be in the range [1, number of data point per MPI rank ({}))!", used_n_neighbors, input_attr.rank_size) };
    }

    // initialize the k-nearest-neighbors data to default values before calculating them
    aos_matrix<index_type> indices{ detail::shape{ input_attr.rank_size, used_n_neighbors } };
    aos_matrix<real_type> distances{ detail::shape{ input_attr.rank_size, used_n_neighbors }, std::numeric_limits<real_type>::max() };

    // calculate start ID
    const index_type base_id = comm_.rank() * input_attr.rank_size;

// fill default values
#pragma omp parallel for collapse(2)
    for (index_type point = 0; point < input_attr.rank_size; ++point) {
        for (index_type nn = 0; nn < used_n_neighbors; ++nn) {
            indices(point, nn) = base_id + point;
        }
    }

    // correctly set default values for dummy points on last MPI rank
    if (comm_.rank() == comm_.size() - 1) {
        const index_type correct_rank_size = input_attr.total_size - ((comm_.size() - 1) * input_attr.rank_size);
        for (index_type point = correct_rank_size; point < input_attr.rank_size; ++point) {
            for (index_type nn = 0; nn < used_n_neighbors; ++nn) {
                indices(point, nn) = base_id + correct_rank_size - 1;
            }
        }
    }

    // perform the k-nearest-neighbors calculation using locality sensitive hashing
    hash_tables_->search_nearest_neighbors(used_n_neighbors, X, indices, distances);

// convert the distances since the sqrt is currently not applied
#pragma omp parallel for collapse(2)
    for (std::size_t row = 0; row < distances.num_rows(); ++row) {
        for (std::size_t col = 0; col < distances.num_cols(); ++col) {
            distances(row, col) = std::sqrt(distances(row, col));
        }
    }

    const auto runtime = mpi_timer.elapsed();
    mpi::detail::log(comm_, "Calculated {} nearest-neighbors in {}.\n\n", used_n_neighbors, runtime);

    // add entry if available
    if (profiler_ != nullptr) {
        profiler_->add_entry("nearest_neighbors", "total_runtime", runtime);
        profiler_->add_entry("nearest_neighbors", "return_distance", return_distances);
        profiler_->add_entry("nearest_neighbors", "n_neighbors", used_n_neighbors);
        profiler_->add_entry("nearest_neighbors", X.get_attributes());
    }

    if (return_distances) {
        return nearest_neighbors_result{ comm_, std::move(X), std::move(indices), std::move(distances), profiler_ };
    }
    return nearest_neighbors_result{ comm_, std::move(X), std::move(indices), profiler_ };
}

}  // namespace sycl_lsh
