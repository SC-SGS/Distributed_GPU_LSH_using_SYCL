/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/nearest_neighbors.hpp"

#include "sycl_lsh/data_attributes.hpp"   // sycl_lsh::data_attributes
#include "sycl_lsh/data_set.hpp"          // sycl_lsh::data_set
#include "sycl_lsh/detail/shape.hpp"      // sycl_lsh::detail::shape
#include "sycl_lsh/hash_tables.hpp"       // sycl_lsh::hash_tables
#include "sycl_lsh/matrix.hpp"            // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/logger.hpp"        // sycl_lsh::mpi::logger
#include "sycl_lsh/mpi/timer.hpp"         // sycl_lsh::mpi::timer
#include "sycl_lsh/options.hpp"           // sycl_lsh::locality_sensitive_hashing_options

#include "fmt/format.h"  // fmt::format

#include <optional>  // std::nullopt
#include <utility>   // std::move

namespace sycl_lsh {

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //

nearest_neighbors::nearest_neighbors(const index_type k, const locality_sensitive_hashing_options &lsh_options, sycl::queue queue, const mpi::communicator &comm, const mpi::logger &logger) :
    queue_{ std::move(queue) },
    comm_{ comm },
    logger_{ logger },
    n_neighbors_{ k },
    lsh_options_{ lsh_options } {
    if (k < 1) {
        throw exception{ fmt::format("k ({}) must be larger than 0!", k) };
    }
}

void nearest_neighbors::fit(data_set X) {
    const mpi::timer mpi_timer{ comm_ };

    // perform some sanity checks
    if (X.attributes().rank_size == 0) {
        throw exception{ "The provided training data set is empty!" };
    }

    // store the provided data set internally (so it can be used in kneighbors if no data set has been provided)
    data_ = std::move(X);

    // create the hash tables
    switch (lsh_options_.hash_function) {
        case hash_function_type::random_projections:
            hash_tables_ = std::make_unique<hash_tables<random_projections>>(lsh_options_, data_, queue_, comm_, logger_);
            break;
        case hash_function_type::entropy_based:
            hash_tables_ = std::make_unique<hash_tables<entropy_based>>(lsh_options_, data_, queue_, comm_, logger_);
            break;
        case hash_function_type::mixed_hash_functions:
            hash_tables_ = std::make_unique<hash_tables<mixed_hash_functions>>(lsh_options_, data_, queue_, comm_, logger_);
            break;
    }

    logger_.log("Fit the nearest-neighbors estimator in {}.\n\n", mpi_timer.elapsed());
}

auto nearest_neighbors::kneighbors_impl(data_set X, const index_type used_n_neighbors, const bool return_distances) const -> results {
    const mpi::timer mpi_timer{ comm_ };

    const data_attributes input_attr = X.attributes();

    // perform some sanity checks
    if (hash_tables_ == nullptr) {
        throw exception{ "Hash tables not initialized yet. Did you call fit()?" };
    }
    if (input_attr.rank_size == 0) {
        throw exception{ "The provided training data set is empty!" };
    }
    if (input_attr.dims != data_.attributes().dims) {
        throw exception{ fmt::format("The number of dimensions in the training data set ({}) and the provided data set ({}) must be the same!", input_attr.dims, data_.attributes().dims) };
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

    logger_.log("Calculated {} nearest-neighbors in {}.\n\n", used_n_neighbors, mpi_timer.elapsed());

    // TODO: currently, returns results per rank -> return results across all ranks doing a gather operation?

    // TODO: change kernel logic to only calculate distances if requested?
    if (return_distances) {
        return { std::move(indices), std::move(distances) };
    }
    return { std::move(indices), std::nullopt };
}

}  // namespace sycl_lsh
