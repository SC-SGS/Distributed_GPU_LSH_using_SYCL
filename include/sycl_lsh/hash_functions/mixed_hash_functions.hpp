/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the mixed hash function as the used LSH hash functions.
 */

#ifndef SYCL_LSH_HASH_FUNCTIONS_MIXED_HASH_FUNCTIONS_HPP
#define SYCL_LSH_HASH_FUNCTIONS_MIXED_HASH_FUNCTIONS_HPP
#pragma once

#include "sycl_lsh/data_attributes.hpp"                // sycl_lsh::data_attributes
#include "sycl_lsh/detail/device_ptr.hpp"              // sycl_lsh::detail::device_ptr
#include "sycl_lsh/detail/hash_combine.hpp"            // sycl_lsh::detail::hash_combine
#include "sycl_lsh/detail/lsh_hash.hpp"                // forward declaration
#include "sycl_lsh/hash_functions/hash_functions.hpp"  // forward declaration
#include "sycl_lsh/mpi/communicator.hpp"               // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/logger.hpp"                     // sycl_lsh::mpi::logger
#include "sycl_lsh/options.hpp"                        // sycl_lsh::locality_sensitive_hashing_options

#include "sycl/sycl.hpp"  // sycl::queue

namespace sycl_lsh {

namespace detail {

/**
 * @brief Specialization of the @ref sycl_lsh::lsh_hash class for the @ref sycl_lsh::mixed_hash_functions class to calculate the
 *        hash value.
 */
template <>
struct lsh_hash<mixed_hash_functions> {
    /**
     * @brief Calculates the hash value of the data @p point in hash table @p hash_tables using mixed hash functions.
     * @param[in] hash_table the provided hash table
     * @param[in] point the provided data point
     * @param[in] data_d the data set
     * @param[in] hash_functions_d the hash functions
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] attr the used @ref sycl_lsh::data_attributes
     * @return the calculated hash value using mixed hash functions (`[[nodiscard]]`)
     *
     * @pre @p hash_table must be in the range `[0, number of hash tables)` (currently disabled).
     * @pre @p hash_function must be in the range `[0, number of hash functions)` (currently disabled).
     */
    [[nodiscard]] hash_value_type operator()(const index_type hash_table, const index_type point, const real_type *data_d, const real_type *hash_functions_d, const locality_sensitive_hashing_options &opt, const data_attributes &attr) const {
        real_type value = 0.0;
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            // calculate hash value using random projections
            real_type hash = hash_functions_d[hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1) + hash_function * (attr.dims + 1) + attr.dims];
            for (index_type dim = 0; dim < attr.dims; ++dim) {
                hash += data_d[point * attr.dims + dim]
                        * hash_functions_d[hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1) + hash_function * (attr.dims + 1) + dim];
            }
            // combine hash values using the entropy-based hash functions
            value += static_cast<hash_value_type>(hash / opt.w)
                     * hash_functions_d[hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1) + opt.num_hash_functions * (attr.dims + 1) + hash_function];
        }
        // calculate final hash value using the cut-off points of the combined hash values
        hash_value_type combined_hash = 0;
        for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
            combined_hash += value > hash_functions_d[hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1) + opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + cop];
        }
        return combined_hash % opt.hash_table_size;
    }
};

}  // namespace detail

/**
 * @brief Class which represents the mixed hash functions used in the LSH algorithm.
 */
class mixed_hash_functions {
  public:
    /**
     * @brief Struct to specify the part of the host buffer when requesting the conversion of a multidimensional index to a
     *        one-dimensional index.
     */
    struct buffer_part {
        /** Calculate conversion only for the random projections part. */
        constexpr static struct hash_functions_t {
        } hash_functions{};

        /** Calculate conversion only for the entropy-based part. */
        constexpr static struct hash_combine_t {
        } hash_combine{};

        /** Calculate conversion only for the cut-off points part. */
        constexpr static struct cut_off_points_t {
        } cut_off_points{};
    };

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::mixed_hash_functions object representing the hash functions used in the LSH algorithm.
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] data the used data stored on the device
     * @param[in] attributes the data's attributes
     * @param[in] queue the SYCL queue to run on
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     */
    mixed_hash_functions(const locality_sensitive_hashing_options &opt, const detail::device_ptr<real_type> &data, data_attributes attributes, sycl::queue &queue, const mpi::communicator &comm, const mpi::logger &logger);

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

#endif  // SYCL_LSH_HASH_FUNCTIONS_MIXED_HASH_FUNCTIONS_HPP
