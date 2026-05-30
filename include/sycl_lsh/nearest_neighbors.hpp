/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements a nearest-neighbors estimator using locality sensitive hashing.
 */

#ifndef SYCL_LSH_NEAREST_NEIGHBORS_HPP
#define SYCL_LSH_NEAREST_NEIGHBORS_HPP
#pragma once

#include "sycl_lsh/constants.hpp"                   // sycl_lsh::index_type, sycl_lsh::real_type
#include "sycl_lsh/data_set.hpp"                    // sycl_lsh::data_set
#include "sycl_lsh/detail/hashing/hash_tables.hpp"  // sycl_lsh::detail::hashing::hash_tables_base
#include "sycl_lsh/matrix.hpp"                      // sycl_lsh::aos_matrix
#include "sycl_lsh/mpi/communicator.hpp"            // sycl_lsh::mpi::communicator
#include "sycl_lsh/options.hpp"                     // sycl_lsh::locality_sensitive_hashing_options, sycl_lsh::detail::has_only_named_args_v

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
     * @brief Small wrapper struct containing the results of a nearest-neighbors search.
     */
    struct results {
        /// Indices of the nearest points in the population matrix.
        aos_matrix<index_type> indices;
        /// Matrix representing the lengths to nearest-neighbors points, only present if return_distance was set to true.
        std::optional<aos_matrix<real_type>> distances;
    };

    /**
     * @brief Construct a new nearest_neighbor estimator.
     * @param[in] k the default number of nearest-neighbors to calculate
     * @param[in] lsh_options the provided options controlling the locality sensitive hashing behavior
     * @param[in] queue the SYCL queue containing the device used to perform the nearest-neighbors search
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     *
     * @throws sycl_lsh::exception if @p k is smaller than `1`
     */
    nearest_neighbors(index_type k, const locality_sensitive_hashing_options &lsh_options, sycl::queue queue, const mpi::communicator &comm);

    /**
     * @brief Fit the nearest neighbors estimator from the training dataset.
     * @param[in] X the training data
     *
     * @throws sycl_lsh::exception if the data set @p X is empty
     */
    void fit(data_set X);

    /**
     * @brief Find the nearest-neighbors of a point using the training data. Returns indices of and distances to the neighbors of each point.
     * @details Calculates the nearest-neighbors for the data set that was also used for the call to fit()
     * @tparam NamedArgs the type of optional named arguments
     * @param[in] named_args the optional named arguments
     * @return the indices of and distances to the neighbors of each point (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if fit() hasn't been called yet
     * @throws sycl_lsh::exception if the data set @p X is empty
     * @throws sycl_lsh::exception if the used fit data set and @p X have different number of dimensions
     * @throws sycl_lsh::exception if @p k is smaller than `1` or larger than the data set's rank_size
     */
    template <typename... NamedArgs, SYCL_LSH_REQUIRES(detail::has_only_named_args_v<NamedArgs...>)>
    [[nodiscard]] results kneighbors(NamedArgs &&...named_args) const {
        return kneighbors(data_, std::forward<NamedArgs>(named_args)...);
    }

    /**
     * @brief Find the nearest-neighbors of all points in @p X Returns indices of and distances to the neighbors of each point.
     * @tparam NamedArgs the type of optional named arguments
     * @param[in] X the data set to calculate the nearest-neighbors for
     * @param[in] named_args the optional named arguments
     * @return the indices of and distances to the neighbors of each point in @p X (`[[nodiscard]]`)
     *
     * @throws sycl_lsh::exception if fit() hasn't been called yet
     * @throws sycl_lsh::exception if the data set @p X is empty
     * @throws sycl_lsh::exception if the used fit data set and @p X have different number of dimensions
     * @throws sycl_lsh::exception if @p k is smaller than `1` or larger than the data set's rank_size
     */
    template <typename... NamedArgs, SYCL_LSH_REQUIRES(detail::has_only_named_args_v<NamedArgs...>)>
    [[nodiscard]] results kneighbors(data_set X, NamedArgs &&...named_args) const {
        // check igor parameter
        const igor::parser parser{ std::forward<NamedArgs>(named_args)... };

        // the default number of nearest-neighbors to calculated is the one provided in the constructor
        index_type used_n_neighbors = n_neighbors_;
        // check whether a different number of nearest-neighbors as been requested
        if constexpr (parser.has(sycl_lsh::n_neighbors)) {
            // update the number of nearest-neighbors
            used_n_neighbors = static_cast<decltype(used_n_neighbors)>(parser(sycl_lsh::n_neighbors));
        }

        // per default, also return the distances
        bool used_return_distances = true;
        // check whether a different number of nearest-neighbors as been requested
        if constexpr (parser.has(sycl_lsh::return_distance)) {
            // update whether distances should be returned or not
            used_return_distances = static_cast<decltype(used_return_distances)>(parser(sycl_lsh::return_distance));
        }

        // forward to actual implementation
        return kneighbors_impl(std::move(X), used_n_neighbors, used_return_distances);
    }

  private:
    // Implementing of the k-nearest-neighbors search.
    [[nodiscard]] results kneighbors_impl(data_set X, index_type used_n_neighbors, bool return_distances) const;

    /// The SYCL queue including the associated device used to perform the nearest-neighbors search.
    sycl::queue queue_;
    /// The associated MPI communicator.
    mpi::communicator comm_;

    /// The number of nearest-neighbors to calculate.
    index_type n_neighbors_;

    /// The data used in the call to fit.
    data_set data_{};

    /// The options controlling the locality sensitive hashing behavior.
    locality_sensitive_hashing_options lsh_options_;
    /// The hash tables used in the locality sensitive hashing algorithm.
    std::unique_ptr<detail::hashing::hash_tables_base> hash_tables_{ nullptr };
};

}  // namespace sycl_lsh

#endif  // SYCL_LSH_NEAREST_NEIGHBORS_HPP
