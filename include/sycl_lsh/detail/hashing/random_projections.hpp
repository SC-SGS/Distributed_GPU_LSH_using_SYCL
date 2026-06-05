/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the random projections hash function as the used LSH hash functions.
 */

#ifndef SYCL_LSH_DETAIL_HASHING_RANDOM_PROJECTIONS_HPP
#define SYCL_LSH_DETAIL_HASHING_RANDOM_PROJECTIONS_HPP
#pragma once

#include "sycl_lsh/data_set.hpp"                     // sycl_lsh::data_set::attributes
#include "sycl_lsh/detail/device_ptr.hpp"            // sycl_lsh::detail::device_ptr
#include "sycl_lsh/detail/hashing/hash_combine.hpp"  // sycl_lsh::detail::hashing::hash_combine
#include "sycl_lsh/detail/hashing/lsh_hash.hpp"      // sycl_lsh::detail::hashing::lsh_hash forward declaration
#include "sycl_lsh/mpi/communicator.hpp"             // sycl_lsh::mpi::communicator
#include "sycl_lsh/options.hpp"                      // sycl_lsh::locality_sensitive_hashing_options

#include "sycl/sycl.hpp"  // sycl::queue

namespace sycl_lsh::detail::hashing {

/**
 * @brief Class which represents the random projections hash functions used in the LSH algorithm.
 */
class random_projections {
  public:
    /**
     * @brief Construct a new @ref sycl_lsh::random_projections object representing the hash functions used in the LSH algorithm.
     * @param[in] opt the used @ref sycl_lsh::locality_sensitive_hashing_options
     * @param[in] data the used data stored on the device
     * @param[in] attributes the data's attributes
     * @param[in] queue the SYCL queue to run on
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     */
    random_projections(const locality_sensitive_hashing_options &opt, const device_ptr<real_type> &data, data_set::attributes attributes, sycl::queue &queue, const mpi::communicator &comm);

    /**
     * @brief Returns the device_ptr wrapping the device memory used in the SYCL kernels.
     * @return the device memory (`[[nodiscard]]`)
     */
    [[nodiscard]] const device_ptr<real_type> &get_device_ptr() const noexcept { return device_ptr_; }

    /**
     * @brief Returns the device_ptr wrapping the device memory used in the SYCL kernels.
     * @return the device memory (`[[nodiscard]]`)
     */
    [[nodiscard]] device_ptr<real_type> &get_device_ptr() noexcept { return device_ptr_; }

  private:
    /// The device buffer.
    device_ptr<real_type> device_ptr_;
};

/**
 * @brief Specialization of the @ref sycl_lsh::detail::hashing::lsh_hash class for the @ref sycl_lsh::detail::hashing::random_projections class to calculate the
 *        hash value.
 */
template <>
struct lsh_hash<random_projections> {
    /**
     * @brief Calculates the hash value of the data @p point in hash table @p hash_tables using random projections.
     * @param[in] hash_table the provided hash table
     * @param[in] point the provided data point
     * @param[in] data_d the data set
     * @param[in] hash_functions_d the hash functions
     * @param[in] opt the used @ref sycl_lsh::locality_sensitive_hashing_options
     * @param[in] attr the used @ref sycl_lsh::data_set::attributes
     * @return the calculated hash value using random projections (`[[nodiscard]]`)
     */
    [[nodiscard]] hash_value_type operator()(const index_type hash_table, const index_type point, const real_type *data_d, const real_type *hash_functions_d, const locality_sensitive_hashing_options &opt, const data_set::attributes &attr) const {
        hash_value_type combined_hash = opt.num_hash_functions;
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            // calculate hash for current hash function
            real_type hash = hash_functions_d[hash_table * opt.num_hash_functions * (attr.dims + 1) + hash_function * (attr.dims + 1) + attr.dims];
            for (index_type dim = 0; dim < attr.dims; ++dim) {
                hash += data_d[point * attr.dims + dim]
                        * hash_functions_d[hash_table * opt.num_hash_functions * (attr.dims + 1) + hash_function * (attr.dims + 1) + dim];
            }
            // combine hashes
            combined_hash = hash_combine(combined_hash, static_cast<hash_value_type>(hash / opt.w));
        }
        return combined_hash % opt.hash_table_size;
    }
};

}  // namespace sycl_lsh::detail::hashing

#endif  // SYCL_LSH_DETAIL_HASHING_RANDOM_PROJECTIONS_HPP
