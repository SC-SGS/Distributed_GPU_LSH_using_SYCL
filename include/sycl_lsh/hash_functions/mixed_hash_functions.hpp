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

#include "sycl_lsh/data_set.hpp"                       // sycl_lsh::data
#include "sycl_lsh/detail/assert.hpp"                  // SYCL_LSH_ASSERT
#include "sycl_lsh/detail/device_ptr.hpp"              // sycl_lsh::detail::device_ptr
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

#include "sycl/sycl.hpp"

#include "mpi.h"  // MPI_Bcast, MPI_Allreduce

#include <random>  // std::mt19937, std::random_device, std::normal_distribution, std::uniform_real_distribution, std::uniform_int_distribution
#include <vector>  // std::vector

namespace sycl_lsh {

namespace detail {

/**
 * @brief Specialization of the @ref sycl_lsh::get_linear_id class for the @ref sycl_lsh::mixed_hash_functions class to convert a
 *        multidimensional index to a one-dimensional one.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 */
template <memory_layout layout>
struct get_linear_id<mixed_hash_functions<layout>> {
    /// The used hash functions type (mixed hash functions for this specialization).
    using hash_function_type = mixed_hash_functions<layout>;

    /**
     * @brief Convert the multidimensional index to a one-dimensional index.
     * @details Only responsible for the random projections hash functions part.
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
    [[nodiscard]] index_type operator()(const index_type hash_table, const index_type hash_function, const index_type dim, const device_accessible_options &opt, const data_attributes &attr, typename hash_function_type::buffer_part::hash_functions_t) const noexcept {  // TODO
        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            const index_type hash_table_offset = hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1);
            return hash_table_offset + hash_function * (attr.dims + 1) + dim;
        } else {
            // Struct of Arrays
            const index_type hash_table_offset = hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1);
            return hash_table_offset + dim * opt.num_hash_functions + hash_function;
        }
    }

    /**
     * @brief Convert the multidimensional index to a one-dimensional index.
     * @details Only responsible for the entropy-based hash functions part for combining the has functions.
     * @param[in] hash_table the requested hash table
     * @param[in] dim the requested dimension of @p hash_function
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] attr the attributes of the used data set
     * @return the one-dimensional index (`[[nodiscard]]`)
     *
     * @pre @p hash_table must be in the range `[0, number of hash tables)` (currently disabled).
     * @pre @p dim must be in the range `[0, number of hash functions)` (currently disabled).
     */
    [[nodiscard]] index_type operator()(const index_type hash_table, const index_type dim, const device_accessible_options &opt, const data_attributes &attr, typename hash_function_type::buffer_part::hash_combine_t) const noexcept {  // TODO
        // no difference between AoS and SoA
        const index_type hash_table_offset = hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1);
        const index_type hash_combine_offset = hash_table_offset + opt.num_hash_functions * (attr.dims + 1);
        return hash_combine_offset + dim;
    }

    /**
     * @brief Convert the multidimensional index to a one-dimensional index.
     * @details Only responsible for the cut-off points to calculate the final hash value.
     * @param[in] hash_table the requested hash table
     * @param[in] dim the requested dimension of @p hash_function
     * @param[in] opt the used @ref sycl_lsh::options
     * @param[in] attr the attributes of the used data set
     * @return the one-dimensional index (`[[nodiscard]]`)
     *
     * @pre @p hash_table must be in the range `[0, number of hash tables)` (currently disabled).
     * @pre @p dim must be in the range `[0, number of cut-off points)` (currently disabled).
     */
    [[nodiscard]] index_type operator()(const index_type hash_table, const index_type dim, const device_accessible_options &opt, const data_attributes &attr, typename hash_function_type::buffer_part::cut_off_points_t) const noexcept {  // TODO
        // no difference between AoS and SoA
        const index_type hash_table_offset = hash_table * (opt.num_hash_functions * (attr.dims + 1) + opt.num_hash_functions + opt.num_cut_off_points - 1);
        const index_type hash_combine_offset = hash_table_offset + opt.num_hash_functions * (attr.dims + 1);
        const index_type cut_off_points_offset = hash_combine_offset + opt.num_hash_functions;
        return cut_off_points_offset + dim;
    }
};

/**
 * @brief Specialization of the @ref sycl_lsh::lsh_hash class for the @ref sycl_lsh::mixed_hash_functions class to calculate the
 *        hash value.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 */
template <memory_layout layout>
struct lsh_hash<mixed_hash_functions<layout>> {
    /// The used hash functions type (mixed hash functions for this specialization).
    using hash_function_type = mixed_hash_functions<layout>;

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
    [[nodiscard]] hash_value_type operator()(const index_type hash_table, const index_type point, const real_type *data_d, const real_type *hash_functions_d, const device_accessible_options &opt, const data_attributes &attr) const {
        // get indexing functions
        const get_linear_id<hash_function_type> get_linear_id_hash_function{};

        real_type value = 0.0;
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            // calculate hash value using random projections
            real_type hash = hash_functions_d[get_linear_id_hash_function(hash_table, hash_function, attr.dims, opt, attr, hash_function_type::buffer_part::hash_functions)];
            for (index_type dim = 0; dim < attr.dims; ++dim) {
                hash += data_d[point * attr.dims + dim]
                        * hash_functions_d[get_linear_id_hash_function(hash_table, hash_function, dim, opt, attr, hash_function_type::buffer_part::hash_functions)];
            }
            // combine hash values using the entropy-based hash functions
            value += static_cast<hash_value_type>(hash / opt.w)
                     * hash_functions_d[get_linear_id_hash_function(hash_table, hash_function, opt, attr, hash_function_type::buffer_part::hash_combine)];
        }
        // calculate final hash value using the cut-off points of the combined hash values
        hash_value_type combined_hash = 0;
        for (index_type cop = 0; cop < opt.num_cut_off_points - 1; ++cop) {
            combined_hash += value > hash_functions_d[get_linear_id_hash_function(hash_table, cop, opt, attr, hash_function_type::buffer_part::cut_off_points)];
        }
        return combined_hash % opt.hash_table_size;
    }
};

}  // namespace detail

/**
 * @brief Class which represents the mixed hash functions used in the LSH algorithm.
 * @tparam layout the @ref sycl_lsh::memory_layout type
 */
template <memory_layout layout>
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
     * @param[in] data the used @ref sycl_lsh::data
     * @param[in] queue the SYCL queue to run on
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     */
    mixed_hash_functions(const device_accessible_options &opt, data_set &data, sycl::queue &queue, const mpi::communicator &comm, const mpi::logger &logger);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                   getter                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Returns the specified @ref sycl_lsh::memory_layout type.
     * @return the @ref sycl_lsh::memory_layout type (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr static memory_layout get_memory_layout() noexcept { return layout; }

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

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <memory_layout layout>
mixed_hash_functions<layout>::mixed_hash_functions(const device_accessible_options &opt, data_set &data, sycl::queue &queue, const mpi::communicator &comm, const mpi::logger &logger) :
    queue_{ queue },
    device_ptr_{ opt.num_hash_tables * opt.num_hash_functions * (data.get_attributes().dims + 1) +  // random projections as hash functions
                     opt.num_hash_tables * (opt.num_hash_functions + opt.num_cut_off_points - 1),   // entropy-based as hash combine
                 queue_ } {
    const mpi::timer mpi_timer{ comm };

    const data_attributes attr = data.get_attributes();

    std::vector<real_type> host_buffer(device_ptr_.size());
    const detail::get_linear_id<mixed_hash_functions> get_linear_id_functor{};

    //
    // CREATE RANDOM PROJECTIONS HASH FUNCTIONS
    //

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

        for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
            for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                const index_type pool_hash_function = rnd_uniform_dist(rnd_uniform_gen);
                for (index_type dim = 0; dim <= attr.dims; ++dim) {
                    host_buffer[get_linear_id_functor(hash_table, hash_function, dim, opt, attr, buffer_part::hash_functions)] = hash_pool[pool_hash_function * (attr.dims + 1) + dim];
                }
            }
        }
    }

    //
    // CREATE ENTROPY-BASED HASH FUNCTIONS
    //

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
        for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
            for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                host_buffer[get_linear_id_functor(hash_table, hash_function, opt, attr, buffer_part::hash_combine)] = rnd_normal_dist(rnd_normal_pool_gen);
            }
        }
    }

    // broadcast random projections hash functions to other MPI ranks
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Bcast(host_buffer.data(), host_buffer.size(), mpi::detail::mpi_datatype<real_type>(), 0, comm.get()));

    // calculate cut-off points
    std::vector<real_type> hash_values(attr.rank_size * opt.num_hash_tables);
    {
        // copy the hash function pool to the device
        detail::device_ptr<real_type> hash_functions_ptr{ host_buffer.size(), queue_ };
        hash_functions_ptr.copy_to_device(host_buffer);

        detail::device_ptr<real_type> hash_values_ptr{ detail::shape{ attr.rank_size, opt.num_hash_tables }, queue_ };

        queue_.submit([&](sycl::handler &cgh) {
            // get device data
            const real_type *data_d = data.get_device_ptr().get();
            const real_type *hash_functions_d = hash_functions_ptr.get();
            real_type *hash_values_d = hash_values_ptr.get();

            // get additional information
            const device_accessible_options options = opt;
            const data_attributes attributes = attr;

            // get get_linear_id functor instantiation
            const detail::get_linear_id<mixed_hash_functions> get_linear_id_hash_functions{};

            cgh.parallel_for(sycl::range<2>{ opt.num_hash_tables, attr.rank_size }, [=](sycl::item<2> item) {
                const index_type idx = item.get_id(1);
                const index_type hash_table = item.get_id(0);

                real_type value = 0.0;
                for (index_type hash_function = 0; hash_function < options.num_hash_functions; ++hash_function) {
                    real_type hash = hash_functions_d[get_linear_id_hash_functions(hash_table, hash_function, attributes.dims, options, attributes, buffer_part::hash_functions)];
                    for (index_type dim = 0; dim < attributes.dims; ++dim) {
                        hash += data_d[idx * attributes.dims + dim]
                                * hash_functions_d[get_linear_id_hash_functions(hash_table, hash_function, dim, options, attributes, buffer_part::hash_functions)];
                    }
                    value += static_cast<hash_value_type>(hash / options.w)
                             * hash_functions_d[get_linear_id_hash_functions(hash_table, hash_function, options, attributes, buffer_part::hash_combine)];
                }
                hash_values_d[hash_table * attr.rank_size + idx] = value;
            });
        });

        // wait until the kernel has finished
        queue_.wait_and_throw();

        // copy the hash values back to the host
        hash_values_ptr.copy_to_host(hash_values);
    }

#pragma omp parallel for
    for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
        // sort hash_values vector in a distributed fashion
        mpi::detail::sort(hash_values.begin() + hash_table * attr.rank_size, hash_values.begin() + (hash_table + 1) * attr.rank_size, comm);

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
                cut_off_points[cop] = hash_values[hash_table * attr.rank_size + cut_off_points_idx[cop] % attr.rank_size];
            }
        }

        // combine to final cut-off points on all MPI ranks
        SYCL_LSH_MPI_ERROR_CHECK(MPI_Allreduce(MPI_IN_PLACE, cut_off_points.data(), cut_off_points.size(), mpi::detail::mpi_datatype<real_type>(), MPI_SUM, comm.get()));

        // copy current cut-off points to hash functions
        for (index_type cop = 0; cop < cut_off_points.size(); ++cop) {
            host_buffer[get_linear_id_functor(hash_table, cop, opt, attr, buffer_part::cut_off_points)] = cut_off_points[cop];
        }
    }

    // broadcast hash function to other MPI ranks
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Bcast(host_buffer.data(), host_buffer.size(), mpi::detail::mpi_datatype<real_type>(), 0, comm.get()));

    // copy the host data to the device
    device_ptr_.copy_to_device(host_buffer);

    logger.log("Created 'mixed_hash_functions' hash functions in {}.\n", mpi_timer.elapsed());
}

}  // namespace sycl_lsh

#endif  // SYCL_LSH_HASH_FUNCTIONS_MIXED_HASH_FUNCTIONS_HPP
