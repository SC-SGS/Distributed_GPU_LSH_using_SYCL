/**
 * @file
 * @author Marcel Breyer
 * @date 2020-12-10
 *
 * @brief Implements the @ref sycl_lsh::hash_tables class representing the used LSH hash tables.
 */

#ifndef SYCL_LSH_HASH_TABLES_HPP
#define SYCL_LSH_HASH_TABLES_HPP
#pragma once

// TODO: move to detailed namespace; move header to hash_functions (rename also?)

#include "sycl_lsh/constants.hpp"                            // sycl_lsh::index_type, sycl_lsh::real_type
#include "sycl_lsh/data_attributes.hpp"                      // sycl_lsh::data_attributes
#include "sycl_lsh/data_set.hpp"                             // sycl_lsh::data_set
#include "sycl_lsh/detail/device_ptr.hpp"                    // sycl_lsh::detail::device_ptr
#include "sycl_lsh/hash_functions/entropy_based.hpp"         // sycl_lsh::entropy_based
#include "sycl_lsh/hash_functions/mixed_hash_functions.hpp"  // sycl_lsh::mixed_hash_functions
#include "sycl_lsh/hash_functions/random_projections.hpp"    // sycl_lsh::random_projections
#include "sycl_lsh/mpi/logger.hpp"                           // sycl_lsh::mpi::logger
#include "sycl_lsh/mpi/timer.hpp"                            // sycl_lsh::mpi::timer
#include "sycl_lsh/options.hpp"                              // sycl_lsh::locality_sensitive_hashing_options

#include "sycl/sycl.hpp"  // sycl::queue

#include <algorithm>  // std::min
#include <array>      // std::array
#include <cmath>      // std::pow, std::floor, std::log2, std::ceil
#include <thread>     // std::thread
#include <utility>    // std::move, std::swap

namespace sycl_lsh {

/**
 * @brief Hash tables base class used to be able to use dynamic polymorphism.
 */
class hash_tables_base {
  public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~hash_tables_base() = default;
    /**
     * @brief Calculate the nearest-neighbors using **Locality Sensitive Hashing**.
     * @param[in] n_neighbors the number of nearest-neighbors to search for
     * @param[in,out] query_data the data to calculate the nearest-neighbors for
     * @param[in,out] indices the calculated nearest-neighbor indices
     * @param[in,out] distances the calculated nearest-neighbor distances
     */
    virtual void search_nearest_neighbors(index_type n_neighbors, data_set &query_data, aos_matrix<index_type> &indices, aos_matrix<real_type> &distances) const = 0;
};

/**
 * @brief Class which represents the hash tables used in the LSH algorithm. Performs the actual calculation of the k-nearest-neighbors.
 * @tparam HashFunction the type of the used hash function: random_projections, entropy_base, or mixed_hash_functions.
 */
template <typename HashFunction>
class hash_tables : public hash_tables_base {
  public:
    /**
     * @brief Constructs a new @ref sycl_lsh::hash_tables object initializing the LSH hash tables.
     * @param[in] lsh_options the used @ref sycl_lsh::options
     * @param[in] data the used @ref sycl_lsh::data representing the used data set
     * @param[in] queue the SYCL queue to run on
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] logger the used @ref sycl_lsh::mpi::logger
     */
    hash_tables(const locality_sensitive_hashing_options &lsh_options, const data_set &data, sycl::queue queue, mpi::communicator comm, const mpi::logger &logger);

    /**
     * @brief Calculate the k-nearest-neighbors using **Locality Sensitive Hashing**, **SYCL** and **MPI**.
     * @param[in] k the number of nearest-neighbors to search for
     * @param[in,out] query_data the data to calculate the nearest-neighbors for
     * @param[in,out] indices the calculated nearest-neighbor indices
     * @param[in,out] distances the calculated nearest-neighbor distances
     */
    void search_nearest_neighbors(index_type k, data_set &query_data, aos_matrix<index_type> &indices, aos_matrix<real_type> &distances) const override;

  private:
    /**
     * @brief Performs the k-nearest-neighbor search given the data set @p data_buffer and already calculate nearest-neighbors @p knns.
     * @param[in] k the number of nearest neighbors to search for
     * @param[in] attr the data attributes
     * @param[in] received_data_ptr the data to perform the nearest-neighbors search on
     * @param[in,out] knn_indices_ptr the (already partially) calculated nearest-neighbor indices
     * @param[in,out] knn_distances_ptr the (already partially) calculated nearest-neighbors distances
     */
    void search_nearest_neighbors_round(index_type k, data_attributes attr, const detail::device_ptr<real_type> &received_data_ptr, detail::device_ptr<index_type> &knn_indices_ptr, detail::device_ptr<real_type> &knn_distances_ptr) const;
    /**
     * @brief Calculate the number of data points assigned to each hash bucket in each hash table.
     * @param[in] attr the data attributes
     * @param[in,out] hash_values_count_ptr the number of data points per hash bucket
     */
    void count_hash_values(data_attributes attr, detail::device_ptr<index_type> &hash_values_count_ptr);
    /**
     * @brief Calculates the offset of each hash bucket in each hash table.
     * @param[in] hash_values_count_ptr the number of data points per hash bucket
     */
    void calculate_offsets(const detail::device_ptr<index_type> &hash_values_count_ptr);
    /**
     * @brief Fill each hash table based on the previously calculated offsets.
     * @param[in] attr the data attributes
     */
    void fill_hash_tables(data_attributes attr);

    /// The associated SYCL queue representing the device to run on.
    mutable sycl::queue queue_;
    /// The associated MPI communicator.
    mpi::communicator comm_;
    /// The associated MPI logger.
    const mpi::logger &logger_;

    /// The SYCL device buffer for the data that is owned by this MPI rank.
    detail::device_ptr<real_type> owning_data_ptr_;

    /// The used options.
    locality_sensitive_hashing_options lsh_options_;
    /// The used has functions.
    std::unique_ptr<HashFunction> hash_functions_;
    /// The SYCL device buffer for the hash functions.
    detail::device_ptr<index_type> hash_tables_ptr_;
    /// The SYCL device buffer for the offsets.
    detail::device_ptr<index_type> offsets_ptr_;
};

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <typename HashFunction>
hash_tables<HashFunction>::hash_tables(const locality_sensitive_hashing_options &lsh_options, const data_set &data, sycl::queue queue, const mpi::communicator comm, const mpi::logger &logger) :
    queue_{ std::move(queue) },
    comm_{ comm },
    logger_{ logger },
    owning_data_ptr_{ data.data().shape(), queue_ },
    lsh_options_{ lsh_options },
    hash_functions_{ nullptr },
    hash_tables_ptr_{ lsh_options_.num_hash_tables * data.attributes().rank_size + BLOCKING_SIZE, queue_ },  // TODO: look at blocking -> change to shape
    offsets_ptr_{ detail::shape{ lsh_options_.num_hash_tables, lsh_options_.hash_table_size + 1 }, queue_ } {
    // log used devices
    logger_.log_on_all("[{}, {}]\n", comm_.rank(), queue_.get_device().get_info<sycl::info::device::name>());
    const mpi::timer mpi_timer{ comm_ };

    // copy the owning data to the device
    owning_data_ptr_.copy_to_device(data.data());

    // create the hash functions (cannot be done earlier since we first need the data on the device)
    hash_functions_ = std::make_unique<HashFunction>(lsh_options_, owning_data_ptr_, data.attributes(), queue_, comm, logger);

    {
        // create temporary buffer to count the occurrence of each hash value
        detail::device_ptr<index_type> hash_values_count_ptr{ detail::shape{ lsh_options_.num_hash_tables, lsh_options_.hash_table_size }, queue_ };

        // count the occurrence of each hash value per hash table
        this->count_hash_values(data.attributes(), hash_values_count_ptr);
        // calculate the offset values
        this->calculate_offsets(hash_values_count_ptr);
    }

    // fill the hash tables based on the previously calculated offsets
    this->fill_hash_tables(data.attributes());

    logger_.log("Created hash tables in {}.\n", mpi_timer.elapsed());
}

template <typename HashFunction>
void hash_tables<HashFunction>::count_hash_values(const data_attributes attr, detail::device_ptr<index_type> &hash_values_count_ptr) {
    const mpi::timer mpi_timer{ comm_ };

    queue_.submit([&](sycl::handler &cgh) {
        // get device data
        index_type *hash_values_count = hash_values_count_ptr.get();
        const real_type *hash_functions = hash_functions_->get_device_ptr().get();
        const real_type *data = owning_data_ptr_.get();

        // get additional information
        const locality_sensitive_hashing_options options = lsh_options_;

        // get hasher functor instantiation
        const detail::lsh_hash<HashFunction> hasher{};

        cgh.parallel_for(sycl::range<2>{ options.num_hash_tables, attr.rank_size }, [=](sycl::item<2> item) {
            const index_type hash_table = item.get_id(0);
            const index_type idx = item.get_id(1);

            const hash_value_type hash_value = hasher(hash_table, idx, data, hash_functions, options, attr);
            detail::atomic_op<index_type>{ hash_values_count[hash_table * options.hash_table_size + hash_value] } += index_type{ 1 };
        });
    });

    // wait until the kernel finished
    queue_.wait_and_throw();

    logger_.log("Counted hash values in {}.\n", mpi_timer.elapsed());
}

template <typename HashFunction>
void hash_tables<HashFunction>::calculate_offsets(const detail::device_ptr<index_type> &hash_values_count_ptr) {
    const mpi::timer mpi_timer{ comm_ };

    queue_.submit([&](sycl::handler &cgh) {
        // get device data
        const index_type *hash_values_count = hash_values_count_ptr.get();
        index_type *offsets = offsets_ptr_.get();

        // get additional information
        const locality_sensitive_hashing_options options = lsh_options_;

        cgh.parallel_for(sycl::range<1>{ options.num_hash_tables }, [=](sycl::item<1> item) {
            const index_type idx = item.get_linear_id();

            // calculate constant offsets
            const index_type hash_value_count_offset = idx * options.hash_table_size;
            const index_type hash_table_offset = idx * (options.hash_table_size + 1);

            // fill the offset values
            for (index_type hash_value = 2; hash_value <= options.hash_table_size; ++hash_value) {
                // calculated modified prefix sum
                offsets[hash_table_offset + hash_value] =
                    offsets[hash_table_offset + hash_value - 1] + hash_values_count[hash_value_count_offset + hash_value - 2];
            }
        });
    });

    // wait until the kernel finished
    queue_.wait_and_throw();

    logger_.log("Calculated offsets in {}.\n", mpi_timer.elapsed());
}

template <typename HashFunction>
void hash_tables<HashFunction>::fill_hash_tables(const data_attributes attr) {
    const mpi::timer mpi_timer{ comm_ };

    queue_.submit([&](sycl::handler &cgh) {
        // get device data
        const real_type *data = owning_data_ptr_.get();
        const real_type *hash_functions = hash_functions_->get_device_ptr().get();
        index_type *offsets = offsets_ptr_.get();
        index_type *hash_tables = hash_tables_ptr_.get();

        // get additional information
        const locality_sensitive_hashing_options options = lsh_options_;
        const index_type base_id = comm_.rank() * attr.rank_size;
        const index_type comm_rank = comm_.rank();
        const index_type comm_size = comm_.size();

        // get hasher functor instantiation
        const detail::lsh_hash<HashFunction> hasher{};

        cgh.parallel_for(sycl::range<2>{ options.num_hash_tables, attr.rank_size }, [=](sycl::item<2> item) {
            const index_type hash_table = item.get_id(0);
            const index_type idx = item.get_id(1);

            // TODO: understand of this could be mitigated by padding
            index_type val = base_id + idx;
            if (comm_rank == comm_size - 1 && hash_table == 0) {
                // set correct values IDs for dummy points
                const index_type correct_rank_size = attr.total_size - ((comm_size - 1) * attr.rank_size);
                if (idx >= correct_rank_size) {
                    val = base_id + correct_rank_size - 1;
                }
            }

            // get hash value
            const hash_value_type hash_value = hasher(hash_table, idx, data, hash_functions, options, attr);
            // update offsets
            const index_type hash_table_idx = detail::atomic_op<index_type>{ offsets[hash_table * (options.hash_table_size + 1) + hash_value + 1] }.fetch_add(index_type{ 1 });
            hash_tables[hash_table * attr.rank_size + hash_table_idx] = val;

            // TODO: understand of this could be mitigated by padding
            // fill additional values needed for blocking
            if (idx == attr.rank_size - 1 && hash_table == 0) {
                for (index_type block = 0; block < BLOCKING_SIZE; ++block) {
                    hash_tables[options.num_hash_tables * attr.rank_size + block] = val;
                }
            }
        });
    });

    // wait until the kernel finished
    queue_.wait_and_throw();

    logger_.log("Filled hash tables in {}.\n", mpi_timer.elapsed());
}

// ---------------------------------------------------------------------------------------------------------- //
//                                       calculate k-nearest-neighbors                                        //
// ---------------------------------------------------------------------------------------------------------- //
template <typename HashFunction>
void hash_tables<HashFunction>::search_nearest_neighbors(const index_type k, data_set &query_data, aos_matrix<index_type> &indices, aos_matrix<real_type> &distances) const {
    detail::device_ptr<real_type> query_data_ptr{ query_data.data().shape(), queue_ };

    detail::device_ptr<index_type> knn_ptr{ indices.shape(), queue_ };
    detail::device_ptr<real_type> knn_dist_ptr{ distances.shape(), queue_ };

    for (int round = 0; round < comm_.size(); ++round) {
        const mpi::timer mpi_round_timer{ comm_ };

        logger_.log("Round {} of {} ... ", round + 1, comm_.size());

        // copy the current data to the device
        query_data_ptr.copy_to_device(query_data.data());

        // create thread to asynchronously perform MPI communication
        std::thread mpi_thread{ [&]() {
            comm_.send_receive_round_robin(query_data.mutable_data());
        } };

        // set the knn data on the device
        knn_ptr.copy_to_device(indices);
        knn_dist_ptr.copy_to_device(distances);

        // calculate k-nearest-neighbors on current MPI rank
        this->search_nearest_neighbors_round(k, query_data.attributes(), query_data_ptr, knn_ptr, knn_dist_ptr);

        // copy the knn data back to the host
        knn_ptr.copy_to_host(indices);
        knn_dist_ptr.copy_to_host(distances);

        // send calculated k-nearest-neighbors and distances to next rank
        comm_.send_receive_round_robin(indices);
        comm_.send_receive_round_robin(distances);
        // wait until all MPI communication has been finished
        mpi_thread.join();
        comm_.barrier();

        logger_.log("finished in {}.\n", mpi_round_timer.elapsed());
    }
}

template <typename HashFunction>
void hash_tables<HashFunction>::search_nearest_neighbors_round(const index_type k, const data_attributes attr, const detail::device_ptr<real_type> &received_data_ptr, detail::device_ptr<index_type> &knn_indices_ptr, detail::device_ptr<real_type> &knn_distances_ptr) const {
    // TODO 2020-10-07 15:52 marcel: check if correct and useful
    const index_type local_mem_size = queue_.get_device().get_info<sycl::info::device::local_mem_size>();
    const index_type max_local_size = local_mem_size / (k * (sizeof(index_type) + sizeof(real_type)));
    const index_type max_work_group_size = queue_.get_device().get_info<sycl::info::device::max_work_group_size>();
    index_type local_size = std::min<index_type>(std::pow(2, std::floor(std::log2(max_local_size))), max_work_group_size);
    if (max_local_size == local_size) {
        local_size /= 2;
    }

    const index_type global_size = static_cast<index_type>(std::ceil(static_cast<double>(attr.rank_size) / static_cast<double>(local_size))) * local_size;

    queue_.submit([&](sycl::handler &cgh) {
        // get device data
        const real_type *data_owned = owning_data_ptr_.get();
        const real_type *data_received = received_data_ptr.get();
        const real_type *hash_functions = hash_functions_->get_device_ptr().get();
        const index_type *offsets = offsets_ptr_.get();
        const index_type *hash_tables = hash_tables_ptr_.get();
        index_type *knn = knn_indices_ptr.get();
        real_type *knn_dist = knn_distances_ptr.get();

        // get additional information
        const locality_sensitive_hashing_options options = lsh_options_;
        const index_type base_id = comm_.rank() * attr.rank_size;

        // get hasher functor instantiation
        const detail::lsh_hash<HashFunction> hasher{};

        // create local memory accessors
        sycl::local_accessor<index_type, 1> knn_local_mem{ sycl::range<1>{ local_size * k }, cgh };
        sycl::local_accessor<real_type, 1> knn_dist_local_mem{ sycl::range<1>{ local_size * k }, cgh };

        const sycl::nd_range<1> execution_range{ sycl::range<1>{ global_size }, sycl::range<1>{ local_size } };

        cgh.parallel_for(execution_range, [=](sycl::nd_item<1> item) {
            const index_type global_idx = item.get_global_linear_id();
            const index_type local_idx = item.get_local_linear_id();

            // immediately return if global_idx is out-of-range
            if (global_idx >= attr.rank_size) {
                return;
            }

            // create work-item local memory
            std::array<index_type, BLOCKING_SIZE> knn_blocked{};
            std::array<real_type, BLOCKING_SIZE> knn_dist_blocked{};

            // initialize local memory arrays
            for (index_type nn = 0; nn < k; ++nn) {
                knn_local_mem[local_idx * k + nn] = knn[global_idx * k + nn];
                knn_dist_local_mem[local_idx * k + nn] = knn_dist[global_idx * k + nn];
            }

            // perform nearest-neighbor search for all hash tables
            for (index_type hash_table = 0; hash_table < options.num_hash_tables; ++hash_table) {
                // calculate hash value (= hash bucket) for current point
                const hash_value_type hash_bucket = hasher(hash_table, global_idx, data_received, hash_functions, options, attr);

                // calculate hash bucket offsets
                const index_type bucket_begin = offsets[hash_table * (options.hash_table_size + 1) + hash_bucket];
                const index_type bucket_end = offsets[hash_table * (options.hash_table_size + 1) + hash_bucket + 1];

                // perform nearest-neighbor search for all data points in the calculated hash bucket
                for (index_type bucket_elem = bucket_begin; bucket_elem < bucket_end; bucket_elem += BLOCKING_SIZE) {
                    // initialize thread local blocking array
                    for (index_type block = 0; block < BLOCKING_SIZE; ++block) {
                        knn_blocked[block] = hash_tables[hash_table * attr.rank_size + bucket_elem + block];
                        knn_dist_blocked[block] = 0.0;
                    }

                    // calculate distances
                    for (index_type block = 0; block < BLOCKING_SIZE; ++block) {
                        for (index_type dim = 0; dim < attr.dims; ++dim) {
                            const real_type x = data_received[global_idx * attr.dims + dim];
                            const real_type y = data_owned[(knn_blocked[block] - base_id) * attr.dims + dim];
                            knn_dist_blocked[block] += (x - y) * (x - y);
                        }
                    }

                    // check candidate function
                    const auto is_candidate = [&](const index_type candidate_idx) {
                        if (candidate_idx - base_id == global_idx) {
                            return false;
                        }
                        for (index_type nn = 0; nn < k; ++nn) {
                            if (knn_local_mem[local_idx * k + nn] == candidate_idx) {
                                return false;
                            }
                        }
                        return true;
                    };

                    // update nearest-neighbors
                    for (index_type block = 0; block < BLOCKING_SIZE; ++block) {
                        if (knn_dist_blocked[block] < knn_dist_local_mem[local_idx * k] && is_candidate(knn_blocked[block])) {
                            knn_local_mem[local_idx * k] = knn_blocked[block];
                            knn_dist_local_mem[local_idx * k] = knn_dist_blocked[block];

                            // ensure that the greatest distance is at pos 0 (bubble-sort)
                            for (index_type nn = 0; nn < k - 1; ++nn) {
                                if (knn_dist_local_mem[local_idx * k + nn] < knn_dist_local_mem[local_idx * k + nn + 1]) {
                                    std::swap(knn_local_mem[local_idx * k + nn], knn_local_mem[local_idx * k + nn + 1]);
                                    std::swap(knn_dist_local_mem[local_idx * k + nn], knn_dist_local_mem[local_idx * k + nn + 1]);
                                }
                            }
                        }
                    }
                }
            }

            // write back to global buffer
            for (index_type nn = 0; nn < k; ++nn) {
                knn[global_idx * k + nn] = knn_local_mem[local_idx * k + nn];
                knn_dist[global_idx * k + nn] = knn_dist_local_mem[local_idx * k + nn];
            }
        });
    });

    // wait until all k-nearest-neighbors were calculated on the current MPI rank
    queue_.wait_and_throw();
}

}  // namespace sycl_lsh

#endif  // SYCL_LSH_HASH_TABLES_HPP
