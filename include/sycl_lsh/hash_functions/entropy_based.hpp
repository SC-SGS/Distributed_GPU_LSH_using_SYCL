/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the entropy based hash function as the used LSH hash functions.
 */

#ifndef SYCL_LSH_HASH_FUNCTIONS_ENTROPY_BASED_HPP
#define SYCL_LSH_HASH_FUNCTIONS_ENTROPY_BASED_HPP
#pragma once

#include "sycl_lsh/data_set.hpp"                       // sycl_lsh::data_set
#include "sycl_lsh/detail/device_ptr.hpp"              // sycl_lsh::detail::device_ptr
#include "sycl_lsh/detail/hash_combine.hpp"            // sycl_lsh::detail::hash_combine
#include "sycl_lsh/detail/lsh_hash.hpp"                // forward declaration
#include "sycl_lsh/hash_functions/hash_functions.hpp"  // forward declaration
#include "sycl_lsh/mpi/communicator.hpp"               // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/logger.hpp"                     // sycl_lsh::mpi::logger
#include "sycl_lsh/options.hpp"                        // sycl_lsh::options

#include "sycl/sycl.hpp"  // sycl::queue

namespace sycl_lsh {

namespace detail {

/**
 * @brief Specialization of the @ref sycl_lsh::lsh_hash class for the @ref sycl_lsh::entropy_based class to calculate the
 *        hash value.
 */
template <>
struct lsh_hash<entropy_based> {
    /**
     * @brief Calculates the hash value of the data @p point in hash table @p hash_tables using entropy based hash functions.
     * @param[in] hash_table the provided hash table
     * @param[in] point the provided data point
     * @param[in] data_d the data set
     * @param[in] hash_functions_d the hash functions
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] attr the used @ref sycl_lsh::data_attributes
     * @return the calculated hash value using entropy based hash functions (`[[nodiscard]]`)
     *
     * @pre @p hash_table must be in the range `[0, number of hash tables)` (currently disabled).
     * @pre @p hash_function must be in the range `[0, number of hash functions)` (currently disabled).
     */
    [[nodiscard]] hash_value_type operator()(const index_type hash_table, const index_type point, const real_type *data_d, const real_type *hash_functions_d, const device_accessible_options &opt, const data_attributes &attr) const {
        hash_value_type combined_hash = opt.num_hash_functions;
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            // calculate dot product for current hash function
            real_type hash = 0.0;
            for (index_type dim = 0; dim < attr.dims; ++dim) {
                hash += data_d[point * attr.dims + dim]
                        * hash_functions_d[hash_table * opt.num_hash_functions * (attr.dims + opt.num_cut_off_points - 1) + hash_function * (attr.dims + opt.num_cut_off_points - 1) + dim];
            }
            // calculate entropy hash for current hash function
            hash_value_type entropy_hash = 0;
            for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
                entropy_hash += hash > hash_functions_d[hash_table * opt.num_hash_functions * (attr.dims + opt.num_cut_off_points - 1) + hash_function * (attr.dims + opt.num_cut_off_points - 1) + attr.dims + cop];
            }
            // combine hashes
            combined_hash = hash_combine(combined_hash, entropy_hash);
        }
        return combined_hash % opt.hash_table_size;
    }
};

}  // namespace detail

/**
 * @brief Class which represents the entropy based hash functions used in the LSH algorithm.
 */
class entropy_based {
  public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::entropy_based object representing the hash functions used in the LSH algorithm.
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] data the used @ref sycl_lsh::data
     * @param[in] queue the SYCL queue to run on
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     */
    entropy_based(const device_accessible_options &opt, data_set &data, sycl::queue &queue, const mpi::communicator &comm, const mpi::logger &logger);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                   getter                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Returns the device_ptr wrapping the device memory used in the SYCL kernels.
     * @return the device memory (`[[nodiscard]]`)
     */
    [[nodiscard]] const detail::device_ptr<real_type> &get_device_ptr() const noexcept { return device_ptr_; }

    /**
     * @brief Returns the device_ptr wrapping the device memory used in the SYCL kernels.
     * @return the device memory (`[[nodiscard]]`)
     */
    [[nodiscard]] detail::device_ptr<real_type> &get_device_ptr() noexcept { return device_ptr_; }

  private:
    /// The associated SYCL queue representing the device to run on.
    sycl::queue &queue_;
    /// The device buffer.
    detail::device_ptr<real_type> device_ptr_;
};

}  // namespace sycl_lsh

#endif  // SYCL_LSH_HASH_FUNCTIONS_ENTROPY_BASED_HPP
