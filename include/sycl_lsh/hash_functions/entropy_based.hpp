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

#include "sycl_lsh/data.hpp"                           // sycl_lsh::data
#include "sycl_lsh/detail/assert.hpp"                  // SYCL_LSH_ASSERT
#include "sycl_lsh/detail/get_linear_id.hpp"           // forward declaration
#include "sycl_lsh/detail/hash_combine.hpp"            // sycl_lsh::detail::hash_combine
#include "sycl_lsh/detail/lsh_hash.hpp"                // forward declaration
#include "sycl_lsh/device_selector.hpp"                // sycl_lsh::device_selector
#include "sycl_lsh/hash_functions/hash_functions.hpp"  // forward declaration
#include "sycl_lsh/memory_layout.hpp"                  // sycl_lsh::memory_layout_type
#include "sycl_lsh/mpi/communicator.hpp"               // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/sort.hpp"                // sycl_lsh::mpi::detail::sort
#include "sycl_lsh/mpi/detail/utility.hpp"             // SYCL_LSH_MPI_ERROR_CHECK
#include "sycl_lsh/mpi/logger.hpp"                     // sycl_lsh::mpi::logger
#include "sycl_lsh/mpi/timer.hpp"                      // sycl_lsh::mpi::timer
#include "sycl_lsh/options.hpp"                        // sycl_lsh::options

#include "mpi.h"      // MPI_Bcast, MPI_Allreduce
#include "sycl/sycl.hpp"  // sycl::buffer, sycl::accessor, sycl::queue

#include <random>  // std::mt19937, std::random_device, std::normal_distribution, std::uniform_real_distribution, std::uniform_int_distribution
#include <vector>  // std::vector

namespace sycl_lsh {

namespace detail {

/**
 * @brief Specialization of the @ref sycl_lsh::get_linear_id class for the @ref sycl_lsh::entropy_based class to convert a
 *        multidimensional index to a one-dimensional one.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 * @tparam Data the @ref sycl_lsh::data type
 */
template <memory_layout layout, typename Data>
struct get_linear_id<entropy_based<layout, Data>> {
    /// The used @ref sycl_lsh::data type.
    using data_type = Data;
    /// The used @ref sycl_lsh::data_attributes type.
    using data_attributes_type = typename data_type::data_attributes_type;

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
    [[nodiscard]] index_type operator()(const index_type hash_table, const index_type hash_function, const index_type dim, const device_accessible_options &opt, const data_attributes_type &attr) const noexcept {  // TODO
        // SYCL_LSH_ASSERT(0 <= hash_table && hash_table < opt.num_hash_tables, "Out-of-bounce access for hash table!");
        // SYCL_LSH_ASSERT(0 <= hash_function && hash_function < opt.hash_pool_size, "Out-of-bounce access for hash function!");
        // SYCL_LSH_ASSERT(0 <= dim && dim < attr.dims, "Out-of-bounce access for dimension!");

        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return hash_table * opt.num_hash_functions * (attr.dims + opt.num_cut_off_points - 1)
                   + hash_function * (attr.dims + opt.num_cut_off_points - 1)
                   + dim;
        } else {
            // Struct of Arrays
            return hash_table * opt.num_hash_functions * (attr.dims + opt.num_cut_off_points - 1)
                   + dim * opt.num_hash_functions
                   + hash_function;
        }
    }
};

/**
 * @brief Specialization of the @ref sycl_lsh::lsh_hash class for the @ref sycl_lsh::entropy_based class to calculate the
 *        hash value.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 * @tparam Data the @ref sycl_lsh::data type
 */
template <memory_layout layout, typename Data>
struct lsh_hash<entropy_based<layout, Data>> {
    /// The used @ref sycl_lsh::data type.
    using data_type = Data;
    /// The used @ref sycl_lsh::data_attributes type.
    using data_attributes_type = typename data_type::data_attributes_type;

    /// The used hash functions type (entropy based for this specialization).
    using hash_function_type = entropy_based<layout, Data>;

    /**
     * @brief Calculates the hash value of the data @p point in hash table @p hash_tables using entropy based hash functions.
     * @tparam AccData the type of the data set `sycl::accessor`
     * @tparam AccHashFunctions the type of the hash functions `sycl::accessor`
     * @param[in] hash_table the provided hash table
     * @param[in] point the provided data point
     * @param[in] acc_data the data set `sycl::accessor`
     * @param[in] acc_hash_functions the hash functions `sycl::accessor`
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] attr the used @ref sycl_lsh::data_attributes
     * @return the calculated hash value using entropy based hash functions (`[[nodiscard]]`)
     *
     * @pre @p hash_table must be in the range `[0, number of hash tables)` (currently disabled).
     * @pre @p hash_function must be in the range `[0, number of hash functions)` (currently disabled).
     */
    template <typename AccData, typename AccHashFunctions>
    [[nodiscard]] hash_value_type operator()(const index_type hash_table, const index_type point, AccData &acc_data, AccHashFunctions &acc_hash_functions, const device_accessible_options &opt, const data_attributes_type &attr) const {  // TODO:
        // SYCL_LSH_ASSERT(0 <= hash_table && hash_table < opt.num_hash_tables, "Out-of-bounce access for hash tables!");
        // SYCL_LSH_ASSERT(0 <= point && point < attr.rank_size, "Out-of-bounce access for data point!");

        // get indexing functions
        const get_linear_id<hash_function_type> get_linear_id_hash_function{};
        const get_linear_id<data_type> get_linear_id_data{};

        hash_value_type combined_hash = opt.num_hash_functions;
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            // calculate dot product for current hash function
            real_type hash = 0.0;
            for (index_type dim = 0; dim < attr.dims; ++dim) {
                hash += acc_data[get_linear_id_data(point, dim, attr)]
                        * acc_hash_functions[get_linear_id_hash_function(hash_table, hash_function, dim, opt, attr)];
            }
            // calculate entropy hash for current hash function
            hash_value_type entropy_hash = 0;
            for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
                entropy_hash += hash > acc_hash_functions[get_linear_id_hash_function(hash_table, hash_function, attr.dims + cop, opt, attr)];
            }
            // combine hashes
            combined_hash = detail::hash_combine(combined_hash, entropy_hash);
        }
        return combined_hash % opt.hash_table_size;
    }
};

}  // namespace detail

/**
 * @brief Class which represents the entropy based hash functions used in the LSH algorithm.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 * @tparam Data the used @ref sycl_lsh::data type
 */
template <memory_layout layout, typename Data>
class entropy_based {
  public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                type aliases                                                //
    // ---------------------------------------------------------------------------------------------------------- //
    /// The type of the @ref sycl_lsh::data object.
    using data_type = Data;
    /// The type of the @ref sycl_lsh::data_attributes object.
    using data_attributes_type = typename data_type::data_attributes_type;

    /// The type of the device buffer used by SYCL.
    using device_buffer_type = sycl::buffer<real_type, 1>;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new @ref sycl_lsh::entropy_based object representing the hash functions used in the LSH algorithm.
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] data the used @ref sycl_lsh::data
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     */
    entropy_based(const device_accessible_options &opt, data_type &data, const mpi::communicator &comm, const mpi::logger &logger);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                   getter                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Returns the specified @ref sycl_lsh::memory_layout type.
     * @return the @ref sycl_lsh::memory_layout type (`[[nodiscard]]`)
     */
    [[nodiscard]] static constexpr memory_layout get_memory_layout() noexcept { return layout; }

    /**
     * @brief Returns the device buffer used in the SYCL kernels.
     * @return the device buffer (`[[nodiscard]]`)
     */
    [[nodiscard]] device_buffer_type &get_device_buffer() noexcept { return device_buffer_; }

  private:
    /// The device buffer.
    device_buffer_type device_buffer_;
};

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout, typename Data>
entropy_based<layout, Data>::entropy_based(const device_accessible_options &opt, data_type &data, const mpi::communicator &comm, const mpi::logger &logger) : device_buffer_(opt.num_hash_tables * opt.num_hash_functions * (data.get_attributes().dims + opt.num_cut_off_points - 1)) {
    const mpi::timer mpi_timer{ comm };

    const data_attributes_type attr = data.get_attributes();

    // create hash pool functions on MPI master rank and distribute to all other ranks
    std::vector<real_type> hash_functions_pool(opt.hash_pool_size * attr.dims);

    const auto get_linear_id_hash_pool = [=](const index_type hash_function, const index_type dim, [[maybe_unused]] const device_accessible_options &option, [[maybe_unused]] const data_attributes_type &attribute) {
        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return hash_function * attribute.dims + dim;
        } else {
            // Struct of Arrays
            return dim * option.hash_pool_size + hash_function;
        }
    };

    if (comm.is_main_rank()) {
// create random generator
#if defined(SYCL_LSH_RANDOM_NUMBERS_DEBUG)
        // don't seed random engine in debug mode
        std::mt19937 rnd_normal_pool_gen{};
#else
        // seed random engine outside debug mode
        std::random_device rnd_pool_device{};
        std::mt19937 rnd_normal_pool_gen{ rnd_pool_device() };
#endif
        std::normal_distribution<real_type> rnd_normal_dist{};

        // fill hash functions
        for (index_type hash_function = 0; hash_function < opt.hash_pool_size; ++hash_function) {
            for (index_type dim = 0; dim < attr.dims; ++dim) {
                hash_functions_pool[get_linear_id_hash_pool(hash_function, dim, opt, attr)] = rnd_normal_dist(rnd_normal_pool_gen);
            }
        }
    }

    // broadcast pool hash functions to other MPI ranks
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Bcast(hash_functions_pool.data(), static_cast<int>(hash_functions_pool.size()), mpi::detail::mpi_datatype<real_type>(), 0, comm.get()));

    std::vector<real_type> cut_off_points_pool(opt.hash_pool_size * (opt.num_cut_off_points - 1));

    // calculate cut-off points
    {
        sycl::queue queue{ device_selector };
        sycl::buffer<real_type, 1> hash_functions_pool_buffer(hash_functions_pool.data(), hash_functions_pool.size());

        std::vector<real_type> hash_values(attr.rank_size);
        for (index_type hash_function = 0; hash_function < opt.hash_pool_size; ++hash_function) {
            {
                sycl::buffer<real_type, 1> hash_values_buffer(hash_values.data(), hash_values.size());
                queue.submit([&](sycl::handler &cgh) {
                    auto acc_data = data.get_device_buffer().template get_access<sycl::access::mode::read>(cgh);
                    auto acc_hash_functions = hash_functions_pool_buffer.template get_access<sycl::access::mode::read>(cgh);
                    auto acc_hash_values = hash_values_buffer.template get_access<sycl::access::mode::discard_write>(cgh);

                    const device_accessible_options options = opt;
                    detail::get_linear_id<data_type> get_linear_id_data{};

                    cgh.parallel_for(sycl::range<>(attr.rank_size), [=](sycl::item<> item) {
                        const index_type idx = item.get_linear_id();

                        real_type value = 0.0;
                        for (index_type dim = 0; dim < attr.dims; ++dim) {
                            value += acc_data[get_linear_id_data(idx, dim, attr)]
                                     * acc_hash_functions[get_linear_id_hash_pool(hash_function, dim, options, attr)];
                        }
                        acc_hash_values[idx] = value;
                    });
                });
            }

            // sort hash_values vector in a distributed fashion
            mpi::detail::sort(hash_values, comm);

            std::vector<real_type> cut_off_points(opt.num_cut_off_points - 1, 0.0);

            // calculate cut-off points indices
            std::vector<index_type> cut_off_points_idx(cut_off_points.size());
            const index_type jump = (attr.rank_size * comm.size()) / opt.num_cut_off_points;
            for (index_type cop = 0; cop < cut_off_points_idx.size(); ++cop) {
                cut_off_points_idx[cop] = (cop + 1) * jump;
            }

            // fill cut-off points which are located on the current MPI rank
            for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
                // check if index belongs to current MPI rank
                if (cut_off_points_idx[cop] >= attr.rank_size * comm.rank() && cut_off_points_idx[cop] < attr.rank_size * (comm.rank() + 1)) {
                    cut_off_points[cop] = hash_values[cut_off_points_idx[cop] % attr.rank_size];
                }
            }

            // combine to final cut-off points on all MPI ranks
            SYCL_LSH_MPI_ERROR_CHECK(MPI_Allreduce(MPI_IN_PLACE, cut_off_points.data(), static_cast<int>(cut_off_points.size()), mpi::detail::mpi_datatype<real_type>(), MPI_SUM, comm.get()));

            // copy current cut-off points to pool
            std::copy(cut_off_points.begin(), cut_off_points.end(), cut_off_points_pool.begin() + hash_function * cut_off_points.size());
        }
    }

    // select actual hash functions
    std::vector<real_type> host_buffer(device_buffer_.size());
    if (comm.is_main_rank()) {
// create random generator
#if defined(SYCL_LSH_RANDOM_NUMBERS_DEBUG)
        // don't seed random engine in debug mode
        std::mt19937 rnd_uniform_gen{};
#else
        // seed random engine outside debug mode
        std::random_device rnd_device{};
        std::mt19937 rnd_uniform_gen{ rnd_device() };
#endif
        std::uniform_int_distribution<index_type> rnd_uniform_dist{ 0, opt.hash_pool_size - 1 };

        detail::get_linear_id<entropy_based> get_linear_id_functor{};

        for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
            for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                const index_type pool_hash_function = rnd_uniform_dist(rnd_uniform_gen);
                for (index_type dim = 0; dim < attr.dims; ++dim) {
                    host_buffer[get_linear_id_functor(hash_table, hash_function, dim, opt, attr)] = hash_functions_pool[get_linear_id_hash_pool(pool_hash_function, dim, opt, attr)];
                }
                for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
                    host_buffer[get_linear_id_functor(hash_table, hash_function, attr.dims + cop, opt, attr)] = cut_off_points_pool[pool_hash_function * (opt.num_cut_off_points - 1) + cop];
                }
            }
        }
    }

    // broadcast hash function to other MPI ranks
    MPI_Bcast(host_buffer.data(), static_cast<int>(host_buffer.size()), mpi::detail::mpi_datatype<real_type>(), 0, comm.get());

    // copy data to device buffer
    auto acc = device_buffer_.template get_access<sycl::access::mode::discard_write>();
    for (index_type i = 0; i < acc.size(); ++i) {
        acc[i] = host_buffer[i];
    }

    logger.log("Created 'entropy_based' hash functions in {}.\n", mpi_timer.elapsed());
}

}  // namespace sycl_lsh

#endif  // SYCL_LSH_HASH_FUNCTIONS_ENTROPY_BASED_HPP
