/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements a nearest-neighbors estimator using Locality Sensitive Hashing (LSH).
 */

#ifndef SYCL_LSH_NEAREST_NEIGHBORS_HPP
#define SYCL_LSH_NEAREST_NEIGHBORS_HPP
#pragma once

#include "sycl_lsh/constants.hpp"                   // sycl_lsh::index_type, sycl_lsh::real_type
#include "sycl_lsh/data_set.hpp"                    // sycl_lsh::data_set
#include "sycl_lsh/detail/hashing/hash_tables.hpp"  // sycl_lsh::detail::hashing::hash_tables_base
#include "sycl_lsh/matrix.hpp"                      // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"            // sycl_lsh::mpi::communicator
#include "sycl_lsh/nearest_neighbors_result.hpp"    // sycl_lsh::nearest_neighbors_result
#include "sycl_lsh/options.hpp"                     // sycl_lsh::locality_sensitive_hashing_options, sycl_lsh::detail::{has_only_named_args_v, sanity_check_locality_sensitive_hashing_options}
#include "sycl_lsh/profiler.hpp"                    // sycl_lsh::profiler

#include "igor/igor.hpp"  // igor::parser

#include <memory>    // std::unique_ptr
#include <optional>  // std::optional
#include <utility>   // std::forward, std::move

namespace sycl_lsh {

/**
 * @brief A class used to perform an unsupervised nearest-neighbors search.
 */
class nearest_neighbors {
  public:
    /**
     * @brief Construct a new nearest_neighbor estimator.
     * @tparam NamedArgs the type of optional named arguments
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] queue the SYCL queue containing the device used to perform the nearest-neighbors search
     * @param[in] named_args the optional named arguments
     *
     * @throws sycl_lsh::exception if the number of nearest-neighbors is smaller than 1
     */
    template <typename... NamedArgs, SYCL_LSH_REQUIRES(detail::has_only_named_args_v<NamedArgs...>)>
    nearest_neighbors(const mpi::communicator comm, sycl::queue queue, NamedArgs &&...named_args) :
        comm_{ comm },
        queue_{ std::move(queue) } {
        // check igor parameter
        const igor::parser parser{ std::forward<NamedArgs>(named_args)... };

        // check whether a different number of nearest-neighbors has been requested
        if constexpr (parser.has(sycl_lsh::n_neighbors)) {
            // update the number of nearest-neighbors
            n_neighbors_ = static_cast<decltype(n_neighbors_)>(parser(sycl_lsh::n_neighbors));
        }

        // check whether a different locality sensitive hashing options has been requested
        if constexpr (parser.has(sycl_lsh::lsh_options)) {
            // update the options
            lsh_options_ = static_cast<decltype(lsh_options_)>(parser(sycl_lsh::lsh_options));
            // perform some sanity checks
            detail::sanity_check_locality_sensitive_hashing_options(lsh_options_);
        }

        // check whether a performance profiler has been provided
        if constexpr (parser.has(sycl_lsh::perf_profiler)) {
            // update the profiler
            profiler_ = static_cast<decltype(profiler_)>(parser(sycl_lsh::perf_profiler));
        }

        // perform some sanity checks
        if (n_neighbors_ < index_type{ 1 }) {
            throw exception{ fmt::format("the number of nearest-neighbors ({}) must be larger than 0!", n_neighbors_) };
        }
    }

    /**
     * @brief Fit the nearest neighbors estimator from the training @ref sycl_lsh::data_set.
     * @param[in] X the training data
     *
     * @throws sycl_lsh::exception if the data set @p X is empty
     */
    void fit(data_set X);

    /**
     * @brief Find the nearest-neighbors of a point using the training data. Returns indices of and distances to the neighbors of each point.
     * @details Calculates the nearest-neighbors for the data set that was also used for the call to @ref sycl_lsh::nearest_neighbors::fit().
     * @tparam NamedArgs the type of optional named arguments
     * @param[in] named_args the optional named arguments
     * @return the indices of and distances to the neighbors of each point (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if @ref sycl_lsh::nearest_neighbors::fit() hasn't been called yet
     * @throws sycl_lsh::exception if the @ref sycl_lsh::data_set @p X is empty
     * @throws sycl_lsh::exception if the used @ref sycl_lsh::nearest_neighbors::fit() @ref sycl_lsh::data_set and @p X have different number of dimensions
     * @throws sycl_lsh::exception if @p k is smaller than `1` or larger than the data set's rank_size
     */
    template <typename... NamedArgs, SYCL_LSH_REQUIRES(detail::has_only_named_args_v<NamedArgs...>)>
    [[nodiscard]] nearest_neighbors_result kneighbors(NamedArgs &&...named_args) const {
        return kneighbors(data_, std::forward<NamedArgs>(named_args)...);
    }

    /**
     * @brief Find the nearest-neighbors of all points in @p X Returns indices of and distances to the neighbors of each point.
     * @tparam NamedArgs the type of optional named arguments
     * @param[in] X the @ref sycl_lsh::data_set to calculate the nearest-neighbors for
     * @param[in] named_args the optional named arguments
     * @return the indices of and distances to the neighbors of each point in @p X (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if @ref sycl_lsh::nearest_neighbors::fit() hasn't been called yet
     * @throws sycl_lsh::exception if the @ref sycl_lsh::data_set @p X is empty
     * @throws sycl_lsh::exception if the used @ref sycl_lsh::nearest_neighbors::fit() @ref sycl_lsh::data_set and @p X have different number of dimensions
     * @throws sycl_lsh::exception if @p k is smaller than `1` or larger than the data set's rank_size
     */
    template <typename... NamedArgs, SYCL_LSH_REQUIRES(detail::has_only_named_args_v<NamedArgs...>)>
    [[nodiscard]] nearest_neighbors_result kneighbors(data_set X, NamedArgs &&...named_args) const {
        // check igor parameter
        const igor::parser parser{ std::forward<NamedArgs>(named_args)... };

        // the default number of nearest-neighbors to calculated is the one provided in the constructor
        index_type used_n_neighbors = n_neighbors_;
        // check whether a different number of nearest-neighbors has been requested
        if constexpr (parser.has(sycl_lsh::n_neighbors)) {
            // update the number of nearest-neighbors
            used_n_neighbors = static_cast<decltype(used_n_neighbors)>(parser(sycl_lsh::n_neighbors));
        }

        // per default, also return the distances
        bool used_return_distances = true;
        // check whether a different number of nearest-neighbors has been requested
        if constexpr (parser.has(sycl_lsh::return_distance)) {
            // update whether distances should be returned or not
            used_return_distances = static_cast<decltype(used_return_distances)>(parser(sycl_lsh::return_distance));
        }

        // forward to actual implementation
        return kneighbors_impl(std::move(X), used_n_neighbors, used_return_distances);
    }

  private:
    /**
     * @brief Find the nearest-neighbors of a point using the training data. Returns indices of and distances to the neighbors of each point.
     * @param[in] X the @ref sycl_lsh::data_set to calculate the nearest-neighbors for
     * @param[in] used_n_neighbors the number of nearest-neighbors to calculate
     * @param[in] return_distances whether the nearest-neighbor distances should be returned
     * @return the indices of and distances to the neighbors of each point (`[[nodiscard]]`)
     */
    [[nodiscard]] nearest_neighbors_result kneighbors_impl(data_set X, index_type used_n_neighbors, bool return_distances) const;

    /// The associated @ref sycl_lsh::mpi::communicator.
    mpi::communicator comm_;
    /// The SYCL queue including the associated device used to perform the nearest-neighbors search.
    sycl::queue queue_;

    /// The number of nearest-neighbors to calculate.
    index_type n_neighbors_{ 5 };

    /// The @ref sycl_lsh::data_set used in the call to @ref sycl_lsh::nearest_neighbors::fit().
    data_set data_{};

    /// The @ref sycl_lsh::locality_sensitive_hashing_options controlling the locality sensitive hashing behavior.
    locality_sensitive_hashing_options lsh_options_{};
    /// The @ref sycl_lsh:detail::hashing::hash_tables: used in the locality sensitive hashing algorithm.
    std::unique_ptr<detail::hashing::hash_tables_base> hash_tables_{ nullptr };

    /// The optional @ref sycl_lsh::profiler.
    std::shared_ptr<profiler> profiler_{ nullptr };
};

}  // namespace sycl_lsh

#endif  // SYCL_LSH_NEAREST_NEIGHBORS_HPP
