/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements the random projections hash function as the used LSH hash functions.
 */

#ifndef SYCL_LSH_HASH_FUNCTIONS_RANDOM_PROJECTIONS_HPP
#define SYCL_LSH_HASH_FUNCTIONS_RANDOM_PROJECTIONS_HPP
#pragma once

#include "sycl_lsh/data.hpp"                           // sycl_lsh::data
#include "sycl_lsh/detail/assert.hpp"                  // SYCL_LSH_ASSERT
#include "sycl_lsh/detail/get_linear_id.hpp"           // forward declaration
#include "sycl_lsh/detail/hash_combine.hpp"            // sycl_lsh::detail::hash_combine
#include "sycl_lsh/detail/lsh_hash.hpp"                // forward declaration
#include "sycl_lsh/hash_functions/hash_functions.hpp"  // forward declaration
#include "sycl_lsh/memory_layout.hpp"                  // sycl_lsh::memory_layout_type
#include "sycl_lsh/mpi/communicator.hpp"               // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/utility.hpp"             // SYCL_LSH_MPI_ERROR_CHECK
#include "sycl_lsh/mpi/logger.hpp"                     // sycl_lsh::mpi::logger
#include "sycl_lsh/mpi/timer.hpp"                      // sycl_lsh::mpi::timer
#include "sycl_lsh/options.hpp"                        // sycl_lsh::options

#include "sycl/sycl.hpp"  // sycl::buffer, sycl::accessor

#include "mpi.h"  // MPI_Bcast

#include <cmath>   // std::abs
#include <random>  // std::mt19937, std::random_device, std::normal_distribution, std::uniform_real_distribution, std::uniform_int_distribution
#include <vector>  // std::vector

namespace sycl_lsh {

namespace detail {
/**
 * @brief Specialization of the @ref sycl_lsh::get_linear_id class for the @ref sycl_lsh::random_projections class to convert a
 *        multidimensional index to a one-dimensional one.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 */
template <memory_layout layout>
struct get_linear_id<random_projections<layout>> {
    /**
     * @brief Convert the multidimensional index to a one-dimensional index.
     * @param[in] hash_table the requested hash table
     * @param[in] hash_function the requested hash function
     * @param[in] dim the requested dimension of @p hash_function
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] attr the attributes of the used data set
     * @return the one-dimensional index (`[[nodiscard]]`)
     *
     * @pre @p hash_table must be in the range `[0, number of hash tables)` (currently disabled).
     * @pre @p hash_function must be in the range `[0, number of hash functions)` (currently disabled).
     * @pre @p dim must be in the range `[0, number of dimensions per data point + 1)` (currently disabled).
     */
    [[nodiscard]] index_type operator()(const index_type hash_table, const index_type hash_function, const index_type dim, const device_accessible_options &opt, const data_attributes &attr) const noexcept {  // TODO options
        // SYCL_LSH_ASSERT(0 <= hash_table && hash_table < opt.num_hash_tables, "Out-of-bounce access for hash table!");
        // SYCL_LSH_ASSERT(0 <= hash_function && hash_function < opt.hash_pool_size, "Out-of-bounce access for hash function!");
        // SYCL_LSH_ASSERT(0 <= dim && dim < attr.dims, "Out-of-bounce access for dimension!");

        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return hash_table * opt.num_hash_functions * (attr.dims + 1) + hash_function * (attr.dims + 1) + dim;
        } else {
            // Struct of Arrays
            return hash_table * opt.num_hash_functions * (attr.dims + 1) + dim * opt.num_hash_functions + hash_function;
        }
    }
};

/**
 * @brief Specialization of the @ref sycl_lsh::lsh_hash class for the @ref sycl_lsh::random_projections class to calculate the
 *        hash value.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 */
template <memory_layout layout>
struct lsh_hash<random_projections<layout>> {
    /// The used hash functions type (random projections for this specialization).
    using hash_function_type = random_projections<layout>;

    /**
     * @brief Calculates the hash value of the data @p point in hash table @p hash_tables using random projections.
     * @tparam AccHashFunctions the type of the hash functions `sycl::accessor`
     * @param[in] hash_table the provided hash table
     * @param[in] point the provided data point
     * @param[in] data_d the data set
     * @param[in] acc_hash_functions the hash functions `sycl::accessor`
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] attr the used @ref sycl_lsh::data_attributes
     * @return the calculated hash value using random projections (`[[nodiscard]]`)
     *
     * @pre @p hash_table must be in the range `[0, number of hash tables)` (currently disabled).
     * @pre @p hash_function must be in the range `[0, number of hash functions)` (currently disabled).
     */
    template <typename AccHashFunctions>
    [[nodiscard]] hash_value_type operator()(const index_type hash_table, const index_type point, const real_type *data_d, AccHashFunctions &acc_hash_functions, const device_accessible_options &opt, const data_attributes &attr) const {  // TODO: replace accessor with USM
        // SYCL_LSH_ASSERT(0 <= hash_table && hash_table < opt.num_hash_tables, "Out-of-bounce access for hash tables!");
        // SYCL_LSH_ASSERT(0 <= point && point < attr.rank_size, "Out-of-bounce access for data point!");

        // get indexing functions
        const get_linear_id<hash_function_type> get_linear_id_hash_function{};
        const get_linear_id<data<layout>> get_linear_id_data{};

        hash_value_type combined_hash = opt.num_hash_functions;
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            // calculate hash for current hash function
            real_type hash = acc_hash_functions[get_linear_id_hash_function(hash_table, hash_function, attr.dims, opt, attr)];
            for (index_type dim = 0; dim < attr.dims; ++dim) {
                hash += data_d[get_linear_id_data(point, dim, attr)]
                        * acc_hash_functions[get_linear_id_hash_function(hash_table, hash_function, dim, opt, attr)];
            }
            // combine hashes
            combined_hash = hash_combine(combined_hash, static_cast<hash_value_type>(hash / opt.w));
        }
        return combined_hash % opt.hash_table_size;
    }
};

}  // namespace detail

/**
 * @brief Class which represents the random projections hash functions used in the LSH algorithm.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 */
template <memory_layout layout>
class random_projections {
  public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                type aliases                                                //
    // ---------------------------------------------------------------------------------------------------------- //
    /// The type of the device buffer used by SYCL.
    using device_buffer_type = sycl::buffer<real_type, 1>;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::random_projections object representing the hash functions used in the LSH algorithm.
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] data the used @ref sycl_lsh::data
     * @param[in] queue the SYCL queue to run on
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     */
    random_projections(const device_accessible_options &opt, const data<layout> &data, sycl::queue &queue, const mpi::communicator &comm, const mpi::logger &logger);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                   getter                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Returns the specified @ref sycl_lsh::memory_layout type.
     * @return the @ref sycl_lsh::memory_layout type (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr static memory_layout get_memory_layout() noexcept { return layout; }

    /**
     * @brief Returns the device buffer used in the SYCL kernels.
     * @return the device buffer (`[[nodiscard]]`)
     */
    [[nodiscard]] device_buffer_type &get_device_buffer() noexcept { return device_buffer_; }

  private:
    /// The associated SYCL queue representing the device to run on.
    sycl::queue &queue_;
    /// The device buffer.
    device_buffer_type device_buffer_;
};

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout>
random_projections<layout>::random_projections(const device_accessible_options &opt, const data<layout> &data, sycl::queue &queue, const mpi::communicator &comm, const mpi::logger &logger) :
    queue_{ queue },
    device_buffer_(opt.num_hash_tables * opt.num_hash_functions * (data.get_attributes().dims + 1)) {
    const mpi::timer mpi_timer{ comm };

    const data_attributes &attr = data.get_attributes();

    std::vector<real_type> host_buffer(device_buffer_.size());

    // create hash pool only on MPI master rank
    if (comm.is_main_rank()) {
// create random generators
#if defined(SYCL_LSH_RANDOM_NUMBERS_DEBUG)
        // don't seed random engine in debug mode
        std::mt19937 rnd_normal_pool_gen{};
        std::mt19937 rnd_uniform_pool_gen{};
#else
        // seed random engine outside debug mode
        std::random_device rnd_pool_device{};
        std::mt19937 rnd_normal_pool_gen{ rnd_pool_device() };
        std::mt19937 rnd_uniform_pool_gen{ rnd_pool_device() };
#endif
        std::normal_distribution<real_type> rnd_normal_pool_dist{};
        std::uniform_real_distribution<real_type> rnd_uniform_pool_dist{ 0, opt.w };

        // fill hash pool
        std::vector<real_type> hash_pool(opt.hash_pool_size * (attr.dims + 1));
        for (index_type hash_function = 0; hash_function < opt.hash_pool_size; ++hash_function) {
            for (index_type dim = 0; dim < attr.dims; ++dim) {
                hash_pool[hash_function * (attr.dims + 1) + dim] = std::abs(rnd_normal_pool_dist(rnd_normal_pool_gen));
            }
            hash_pool[hash_function * (attr.dims + 1) + attr.dims] = rnd_uniform_pool_dist(rnd_uniform_pool_gen);
        }

// select actual hash functions
#if defined(SYCL_LSH_RANDOM_NUMBERS_DEBUG)
        // don't seed random engine in debug mode
        std::mt19937 rnd_uniform_gen{};
#else
        // seed random engine outside debug mode
        std::random_device rnd_device{};
        std::mt19937 rnd_uniform_gen{ rnd_device() };
#endif
        std::uniform_int_distribution<index_type> rnd_uniform_dist{ 0, opt.hash_pool_size - 1 };

        const detail::get_linear_id<random_projections> get_linear_id_functor{};

        for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
            for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                const index_type pool_hash_function = rnd_uniform_dist(rnd_uniform_gen);
                for (index_type dim = 0; dim <= attr.dims; ++dim) {
                    host_buffer[get_linear_id_functor(hash_table, hash_function, dim, opt, attr)] = hash_pool[pool_hash_function * (attr.dims + 1) + dim];
                }
            }
        }
    }

    // broadcast hash functions to other MPI ranks
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Bcast(host_buffer.data(), host_buffer.size(), mpi::detail::mpi_datatype<real_type>(), 0, comm.get()));

    // copy data to device buffer
    auto acc = device_buffer_.template get_access<sycl::access::mode::discard_write>();
    for (index_type i = 0; i < acc.size(); ++i) {
        acc[i] = host_buffer[i];
    }

    logger.log("Created 'random_projections' hash functions in {}.\n", mpi_timer.elapsed());
}

}  // namespace sycl_lsh

#endif  // SYCL_LSH_HASH_FUNCTIONS_RANDOM_PROJECTIONS_HPP
