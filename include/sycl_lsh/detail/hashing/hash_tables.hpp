/**
 * @file
 * @author Marcel Breyer
 * @date 2020-12-10
 *
 * @brief Implements the @ref sycl_lsh::detail::hashing::hash_tables class representing the used LSH hash tables.
 */

#ifndef SYCL_LSH_DETAIL_HASHING_HASH_TABLES_HPP
#define SYCL_LSH_DETAIL_HASHING_HASH_TABLES_HPP
#pragma once

#include "sycl_lsh/constants.hpp"                            // sycl_lsh::index_type, sycl_lsh::real_type
#include "sycl_lsh/data_set.hpp"                             // sycl_lsh::data_set
#include "sycl_lsh/detail/device_ptr.hpp"                    // sycl_lsh::device_ptr
#include "sycl_lsh/detail/hashing/entropy_based.hpp"         // sycl_lsh::detail::hashing::entropy_based
#include "sycl_lsh/detail/hashing/mixed_hash_functions.hpp"  // sycl_lsh::detail::hashing::mixed_hash_functions
#include "sycl_lsh/detail/hashing/random_projections.hpp"    // sycl_lsh::detail::hashing::random_projections
#include "sycl_lsh/mpi/detail/logging.hpp"                   // sycl_lsh::mpi::detail::{log, log_from_all}
#include "sycl_lsh/mpi/detail/math.hpp"                      // sycl_lsh::mpi::detail::elementwise_sum_inplace_main
#include "sycl_lsh/mpi/detail/timer.hpp"                     // sycl_lsh::mpi::detail::timer
#include "sycl_lsh/options.hpp"                              // sycl_lsh::locality_sensitive_hashing_options, sycl_lsh::output_with_prefix
#include "sycl_lsh/profiler.hpp"                             // sycl_lsh::profiler

#include "sycl/sycl.hpp"  // sycl::queue

#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <algorithm>  // std::min
#include <array>      // std::array
#include <cmath>      // std::pow, std::floor, std::log2, std::ceil
#include <thread>     // std::thread
#include <utility>    // std::move, std::swap

namespace sycl_lsh::detail::hashing {

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
     * @param[in,out] query_data the @ref sycl_lsh::data_set to calculate the nearest-neighbors for
     * @param[in,out] indices the calculated nearest-neighbor indices
     * @param[in,out] distances the calculated nearest-neighbor distances
     */
    virtual void search_nearest_neighbors(index_type n_neighbors, data_set &query_data, aos_matrix<index_type> &indices, aos_matrix<real_type> &distances) const = 0;
};

/**
 * @brief Class which represents the hash tables used in the LSH algorithm. Performs the actual calculation of the k nearest-neighbors.
 * @tparam HashFunction the type of the used hash function: @ref sycl_lsh::detail::hashing::random_projections,
 *                      @ref sycl_lsh::detail::hashing::entropy_based, or @ref sycl_lsh::detail::hashing::mixed_hash_functions.
 */
template <typename HashFunction>
class hash_tables : public hash_tables_base {
  public:
    /**
     * @brief Constructs a new @ref sycl_lsh::detail::hashing::hash_tables object initializing the LSH hash tables.
     * @param[in] work_group_size the SYCL work-group size for the main kernel(s)
     * @param[in] options the used @ref sycl_lsh::locality_sensitive_hashing_options
     * @param[in] data the used @ref sycl_lsh::data_set representing the used data
     * @param[in] queue the SYCL queue to run on
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in] profiler the performance profiler used to log runtime information, if requested
     */
    hash_tables(std::size_t work_group_size, const locality_sensitive_hashing_options &options, const data_set &data, sycl::queue queue, mpi::communicator comm, std::shared_ptr<profiler> profiler);

    /**
     * @brief Calculate the k nearest-neighbors using **Locality Sensitive Hashing**, **SYCL** and **MPI**.
     * @param[in] k the number of nearest-neighbors to search for
     * @param[in,out] query_data the @ref sycl_lsh::data_set to calculate the nearest-neighbors for
     * @param[in,out] indices the calculated nearest-neighbor indices
     * @param[in,out] distances the calculated nearest-neighbor distances
     */
    void search_nearest_neighbors(index_type k, data_set &query_data, aos_matrix<index_type> &indices, aos_matrix<real_type> &distances) const override;

  private:
    /**
     * @brief Performs the k nearest-neighbor search given the data set @p data_buffer and already calculate nearest-neighbors.
     * @param[in] round the current round
     * @param[in] k the number of nearest neighbors to search for
     * @param[in] attr the @ref sycl_lsh::data_set::attributes of the query @ref sycl_lsh::data_set
     * @param[in] received_data_ptr the data to perform the nearest-neighbors search on
     * @param[in,out] knn_indices_ptr the (already partially) calculated nearest-neighbor indices
     * @param[in,out] knn_distances_ptr the (already partially) calculated nearest-neighbors distances
     * @param[in, out] knn_calculation_count_ptr the (already partially) counted number of nearest-neighbor calculations
     * @return the kernel execution time in milliseconds (`[[nodiscard]]`)
     */
    [[nodiscard]] std::chrono::milliseconds search_nearest_neighbors_round(int round, index_type k, data_set::attributes attr, const device_ptr<real_type> &received_data_ptr, device_ptr<index_type> &knn_indices_ptr, device_ptr<real_type> &knn_distances_ptr, device_ptr<index_type> &knn_calculation_count_ptr) const;
    /**
     * @brief Calculate the number of data points assigned to each hash bucket in each hash table.
     * @param[in] attr the @ref sycl_lsh::data_set::attributes of the query @ref sycl_lsh::data_set
     * @param[in,out] hash_values_count_ptr the number of data points per hash bucket
     */
    void count_hash_values(data_set::attributes attr, device_ptr<index_type> &hash_values_count_ptr);
    /**
     * @brief Calculates the offset of each hash bucket in each hash table.
     * @param[in] hash_values_count_ptr the number of data points per hash bucket
     */
    void calculate_offsets(const device_ptr<index_type> &hash_values_count_ptr);
    /**
     * @brief Fill each hash table based on the previously calculated offsets.
     * @param[in] attr the @ref sycl_lsh::data_set::attributes of the query @ref sycl_lsh::data_set
     */
    void fill_hash_tables(data_set::attributes attr);

    /// The associated SYCL queue representing the device to run on.
    mutable sycl::queue queue_;
    /// The associated @ref sycl_lsh::mpi::communicator.
    mpi::communicator comm_;

    /// The SYCL device buffer for the data that is owned by this MPI rank.
    device_ptr<real_type> owning_data_ptr_;

    /// The SYCL work-group size for the main kernel(s).
    std::size_t work_group_size_;
    /// The used LSH hashing related @ref sycl_lsh::locality_sensitive_hashing_options.
    locality_sensitive_hashing_options lsh_options_;
    /// The used has functions.
    std::unique_ptr<HashFunction> hash_functions_;
    /// The SYCL device buffer for the hash functions.
    device_ptr<index_type> hash_tables_ptr_;
    /// The SYCL device buffer for the offsets.
    device_ptr<index_type> offsets_ptr_;

    /// The optional @ref sycl_lsh::profiler.
    std::shared_ptr<profiler> profiler_{ nullptr };
};

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
template <typename HashFunction>
hash_tables<HashFunction>::hash_tables(const std::size_t work_group_size, const locality_sensitive_hashing_options &options, const data_set &data, sycl::queue queue, const mpi::communicator comm, std::shared_ptr<profiler> profiler) :
    queue_{ std::move(queue) },
    comm_{ comm },
    owning_data_ptr_{ data.data<soa_matrix>().shape(), queue_ },
    work_group_size_{ work_group_size },
    lsh_options_{ options },
    hash_functions_{ nullptr },
    hash_tables_ptr_{ lsh_options_.num_hash_tables * data.get_attributes().rank_size + BLOCKING_SIZE, queue_ },
    offsets_ptr_{ shape{ lsh_options_.num_hash_tables, lsh_options_.hash_table_size + 1 }, queue_ },
    profiler_{ std::move(profiler) } {
    const mpi::detail::timer mpi_timer{ comm_ };

    // add event if available
    if (profiler_ != nullptr) {
        profiler_->add_event("hash_table_create_start");
    }

    // log used devices from all MPI ranks
    const std::vector<std::string> device_names = comm_.gather(fmt::format(queue_.get_device().get_info<sycl::info::device::name>()));
    mpi::detail::log(comm_, "Using the following device(s) for the nearest-neighbor calculation:\n");
    for (std::size_t i = 0; i < device_names.size(); ++i) {
        mpi::detail::log(comm_, "  - [rank {}, {}]\n", i, device_names[i]);
    }
    mpi::detail::log(comm_, "\n");

    // copy the owning data to the device
    owning_data_ptr_.copy_to_device(data.data<soa_matrix>());

    {
        const mpi::detail::timer mpi_timer_hash_functions{ comm_ };

        // add event if available
        if (profiler_ != nullptr) {
            profiler_->add_event("hash_function_create_start");
        }

        // create the hash functions (cannot be done earlier since we first need the data on the device)
        hash_functions_ = std::make_unique<HashFunction>(lsh_options_, owning_data_ptr_, data.get_attributes(), queue_, comm);

        // add event and entry if available
        if (profiler_ != nullptr) {
            profiler_->add_event("hash_function_create_end");
            profiler_->add_entry("fit", "hash_function_create", mpi_timer_hash_functions.elapsed());
        }
    }

    {
        // create temporary buffer to count the occurrence of each hash value
        device_ptr<index_type> hash_values_count_ptr{ shape{ lsh_options_.num_hash_tables, lsh_options_.hash_table_size }, queue_ };

        // count the occurrence of each hash value per hash table
        this->count_hash_values(data.get_attributes(), hash_values_count_ptr);
        // calculate the offset values
        this->calculate_offsets(hash_values_count_ptr);
    }

    // fill the hash tables based on the previously calculated offsets
    this->fill_hash_tables(data.get_attributes());

    // after creating the hash tables, convert the owning data from SoA to AoS layout since this layout is more beneficial in the search_nearest_neighbors kernel
    owning_data_ptr_.copy_to_device(data.data<aos_matrix>());

    // add entry if available
    if (profiler_ != nullptr) {
        profiler_->add_event("hash_table_create_end");
        profiler_->add_entry("fit", "hash_table_create", mpi_timer.elapsed());
        profiler_->add_entry("backend", "device", device_names);
    }
}

template <typename HashFunction>
void hash_tables<HashFunction>::count_hash_values(const data_set::attributes attr, device_ptr<index_type> &hash_values_count_ptr) {
    const mpi::detail::timer mpi_timer{ comm_ };

    // add event if available
    if (profiler_ != nullptr) {
        profiler_->add_event("count_hash_values_start");
    }

    queue_.submit([&](sycl::handler &cgh) {
        // get device data
        index_type *hash_values_count = hash_values_count_ptr.get();
        const real_type *hash_functions = hash_functions_->get_device_ptr().get();
        const real_type *data = owning_data_ptr_.get();

        // get additional information
        const locality_sensitive_hashing_options options = lsh_options_;

        // get hasher functor instantiation
        const lsh_hash<HashFunction> hasher{};

        cgh.parallel_for(sycl::range<2>{ options.num_hash_tables, attr.rank_size }, [=](sycl::item<2> item) {
            const index_type hash_table = item.get_id(0);
            const index_type idx = item.get_id(1);

            const hash_value_type hash_value = hasher(hash_table, idx, data, hash_functions, options, attr);
            atomic_op<index_type>{ hash_values_count[hash_table * options.hash_table_size + hash_value] } += index_type{ 1 };
        });
    });

    // wait until the kernel finished
    queue_.wait_and_throw();

#if defined(SYCL_LSH_HASH_VALUE_DISTRIBUTION_DEBUG)
    // copy the data to the host
    std::vector<index_type> hash_values_count(hash_values_count_ptr.size());
    hash_values_count_ptr.copy_to_host(hash_values_count);

    // collect the data from the other MPI ranks
    mpi::detail::elementwise_sum_inplace_main(hash_values_count, comm_);

    if (comm_.is_main_rank()) {
        // output the count values to a simple .csv file only on the MPI main rank
        std::ofstream out{ "hash_value_distribution.csv" };
        if (!out.good()) {
            throw exception{ "Couldn't create the hash value distribution .csv file!" };
        }

        // add metadata as comment
        output_with_prefix(out, lsh_options_, "# ");
        out << fmt::format("# num_data_points: {}\n", attr.total_size);

        // add the distributions
        for (std::size_t hash_table = 0; hash_table < lsh_options_.num_hash_tables; ++hash_table) {
            out << fmt::format("{},{}", hash_table, fmt::join(hash_values_count.begin() + hash_table * lsh_options_.hash_table_size, hash_values_count.begin() + (hash_table + 1) * lsh_options_.hash_table_size, ",")) << std::endl;
        }
    }
#endif

    // add event and entry if available
    if (profiler_ != nullptr) {
        profiler_->add_event("count_hash_values_end");
        profiler_->add_entry("fit", "count_hash_values", mpi_timer.elapsed());
    }
}

template <typename HashFunction>
void hash_tables<HashFunction>::calculate_offsets(const device_ptr<index_type> &hash_values_count_ptr) {
    const mpi::detail::timer mpi_timer{ comm_ };

    // add event if available
    if (profiler_ != nullptr) {
        profiler_->add_event("calculate_offsets_start");
    }

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

    // add event and entry if available
    if (profiler_ != nullptr) {
        profiler_->add_event("calculate_offsets_end");
        profiler_->add_entry("fit", "calculate_offsets", mpi_timer.elapsed());
    }
}

template <typename HashFunction>
void hash_tables<HashFunction>::fill_hash_tables(const data_set::attributes attr) {
    const mpi::detail::timer mpi_timer{ comm_ };

    // add event if available
    if (profiler_ != nullptr) {
        profiler_->add_event("fill_hash_tables_start");
    }

    const index_type comm_rank = comm_.rank();
    const index_type comm_size = comm_.size();

    queue_.submit([&](sycl::handler &cgh) {
        // get device data
        const real_type *data = owning_data_ptr_.get();
        const real_type *hash_functions = hash_functions_->get_device_ptr().get();
        index_type *offsets = offsets_ptr_.get();
        index_type *hash_tables = hash_tables_ptr_.get();

        // get additional information
        const locality_sensitive_hashing_options options = lsh_options_;
        const index_type base_id = comm_.rank() * attr.rank_size;

        // get hasher functor instantiation
        const lsh_hash<HashFunction> hasher{};

        cgh.parallel_for(sycl::range<2>{ options.num_hash_tables, attr.rank_size }, [=](sycl::item<2> item) {
            const index_type hash_table = item.get_id(0);
            const index_type idx = item.get_id(1);

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
            const index_type hash_table_idx = atomic_op<index_type>{ offsets[hash_table * (options.hash_table_size + 1) + hash_value + 1] }.fetch_add(index_type{ 1 });
            hash_tables[hash_table * attr.rank_size + hash_table_idx] = val;

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

    // add event and entry if available
    if (profiler_ != nullptr) {
        profiler_->add_event("fill_hash_tables_end");
        profiler_->add_entry("fit", "fill_hash_tables", mpi_timer.elapsed());
    }
}

// ---------------------------------------------------------------------------------------------------------- //
//                                       calculate k-nearest-neighbors                                        //
// ---------------------------------------------------------------------------------------------------------- //
template <typename HashFunction>
void hash_tables<HashFunction>::search_nearest_neighbors(const index_type k, data_set &query_data, aos_matrix<index_type> &indices, aos_matrix<real_type> &distances) const {
    device_ptr<real_type> query_data_ptr{ query_data.data<soa_matrix>().shape(), queue_ };

    // add event if available
    if (profiler_ != nullptr) {
        profiler_->add_event("search_nearest_neighbors_start");
    }

    // create a vector containing all round runtimes
    std::vector<std::chrono::milliseconds> round_runtimes{};
    std::vector<std::chrono::milliseconds> round_kernel_runtimes{};

    device_ptr<index_type> knn_ptr{ indices.shape(), queue_ };
    device_ptr<real_type> knn_dist_ptr{ distances.shape(), queue_ };

#if defined(SYCL_LSH_NEAREST_NEIGHBOR_SEARCH_DISTRIBUTION_DEBUG)
    aos_matrix<index_type> knn_search_count{ shape{ lsh_options_.num_hash_tables, query_data.get_attributes().rank_size } };
    device_ptr<index_type> knn_search_count_ptr{ knn_search_count.shape(), queue_ };
#else
    // create an empty device_ptr
    // only serves as dummy for the function call if SYCL_LSH_NEAREST_NEIGHBOR_SEARCH_DISTRIBUTION_DEBUG is not set
    device_ptr<index_type> knn_search_count_ptr{};
#endif

    for (int round = 0; round < comm_.size(); ++round) {
        const mpi::detail::timer mpi_round_timer{ comm_ };

        mpi::detail::log(comm_, "Round {} of {} ... ", round + 1, comm_.size());

        // copy the current data to the device
        query_data_ptr.copy_to_device(query_data.data<soa_matrix>());

        // create thread to asynchronously perform MPI communication
        std::thread mpi_thread{ [&]() {
            comm_.send_receive_round_robin(query_data.mutable_data<soa_matrix>());
        } };

        // set the knn data on the device
        knn_ptr.copy_to_device(indices);
        knn_dist_ptr.copy_to_device(distances);
#if defined(SYCL_LSH_NEAREST_NEIGHBOR_SEARCH_DISTRIBUTION_DEBUG)
        knn_search_count_ptr.copy_to_device(knn_search_count);
#endif

        // calculate k-nearest-neighbors on current MPI rank
        round_kernel_runtimes.push_back(this->search_nearest_neighbors_round(round, k, query_data.get_attributes(), query_data_ptr, knn_ptr, knn_dist_ptr, knn_search_count_ptr));

        // copy the knn data back to the host
        knn_ptr.copy_to_host(indices);
        knn_dist_ptr.copy_to_host(distances);
#if defined(SYCL_LSH_NEAREST_NEIGHBOR_SEARCH_DISTRIBUTION_DEBUG)
        knn_search_count_ptr.copy_to_host(knn_search_count);
#endif

        // send calculated k-nearest-neighbors and distances to next rank
        comm_.send_receive_round_robin(indices);
        comm_.send_receive_round_robin(distances);
#if defined(SYCL_LSH_NEAREST_NEIGHBOR_SEARCH_DISTRIBUTION_DEBUG)
        comm_.send_receive_round_robin(knn_search_count);
#endif
        // wait until all MPI communication has been finished
        mpi_thread.join();
        comm_.barrier();

        const auto runtime = mpi_round_timer.elapsed();
        mpi::detail::log(comm_, "finished in {}.\n", runtime);
        round_runtimes.push_back(runtime);
    }

#if defined(SYCL_LSH_NEAREST_NEIGHBOR_SEARCH_DISTRIBUTION_DEBUG)
    // NOTE: hacky, but works; a simple gather on the matrix data does not work due to the memory layout
    // gather the string for each MPI rank
    std::vector<std::vector<std::string>> hash_table_partial_output(knn_search_count.num_rows());
    for (std::size_t hash_table = 0; hash_table < knn_search_count.num_rows(); ++hash_table) {
        const std::string partial_string_for_rank = fmt::format("{}", fmt::join(knn_search_count.data() + hash_table * knn_search_count.num_cols(), knn_search_count.data() + (hash_table + 1) * knn_search_count.num_cols(), ","));
        hash_table_partial_output[hash_table] = comm_.gather(partial_string_for_rank);
    }

    // on the MPI main rank, collect the strings and write them to the file
    if (comm_.is_main_rank()) {
        // output the nearest-neighbor calculation count to a simple .csv file only on the MPI main rank
        std::ofstream out{ "nearest_neighbor_calculation_count.csv" };
        if (!out.good()) {
            throw exception{ "Couldn't create the nearest-neighbor calculation count .csv file!" };
        }

        // add metadata as comment
        output_with_prefix(out, lsh_options_, "# ");
        out << fmt::format("# num_data_points: {}\n", query_data.get_attributes().total_size);
        out << fmt::format("# BLOCKING_SIZE: {}\n", BLOCKING_SIZE);

        // add the search counts
        for (std::size_t hash_table = 0; hash_table < hash_table_partial_output.size(); ++hash_table) {
            out << fmt::format("{},{}", hash_table, fmt::join(hash_table_partial_output[hash_table].begin(), hash_table_partial_output[hash_table].end(), ",")) << std::endl;
        }
    }
#endif

    // add event and entry if available
    if (profiler_ != nullptr) {
        profiler_->add_event("search_nearest_neighbors_end");
        profiler_->add_entry("nearest_neighbors", "num_rounds", comm_.size());
        profiler_->add_entry("nearest_neighbors", "rounds", round_runtimes);
        profiler_->add_entry("nearest_neighbors", "kernel_rounds", round_kernel_runtimes);
    }
}

template <typename HashFunction>
std::chrono::milliseconds hash_tables<HashFunction>::search_nearest_neighbors_round(const int round, const index_type k, const data_set::attributes attr, const device_ptr<real_type> &received_data_ptr, device_ptr<index_type> &knn_indices_ptr, device_ptr<real_type> &knn_distances_ptr, device_ptr<index_type> &knn_calculation_count_ptr) const {
    const mpi::detail::timer mpi_timer{ comm_ };

    // add event if available
    if (profiler_ != nullptr) {
        profiler_->add_event(fmt::format("search_nearest_neighbors_round_{}_start", round));
    }

    const index_type global_size = static_cast<index_type>(std::ceil(static_cast<double>(attr.rank_size) / static_cast<double>(work_group_size_))) * work_group_size_;

    queue_.submit([&](sycl::handler &cgh) {
        // get device data
        const real_type *data_owned = owning_data_ptr_.get();  // NOTE: in AoS layout now!!!
        const real_type *data_received = received_data_ptr.get();
        const real_type *hash_functions = hash_functions_->get_device_ptr().get();
        const index_type *offsets = offsets_ptr_.get();
        const index_type *hash_tables = hash_tables_ptr_.get();
        index_type *knn = knn_indices_ptr.get();
        real_type *knn_dist = knn_distances_ptr.get();
        index_type *knn_calculation_count = knn_calculation_count_ptr.get();

        // get additional information
        const locality_sensitive_hashing_options options = lsh_options_;
        const index_type base_id = comm_.rank() * attr.rank_size;

        // get hasher functor instantiation
        const lsh_hash<HashFunction> hasher{};

        // create local memory accessors
        sycl::local_accessor<index_type, 2> knn_local_mem{ sycl::range<2>{ work_group_size_, k }, cgh };
        sycl::local_accessor<real_type, 2> knn_dist_local_mem{ sycl::range<2>{ work_group_size_, k }, cgh };

        const sycl::nd_range<1> execution_range{ sycl::range<1>{ global_size }, sycl::range<1>{ work_group_size_ } };

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

            // check candidate function
            const auto is_candidate = [&](const index_type candidate_idx) {
                if (candidate_idx - base_id == global_idx) {
                    return false;
                }
                for (index_type nn = 0; nn < k; ++nn) {
                    if (knn_local_mem[local_idx][nn] == candidate_idx) {
                        return false;
                    }
                }
                return true;
            };

            // initialize local memory arrays
            for (index_type nn = 0; nn < k; ++nn) {
                knn_local_mem[local_idx][nn] = knn[global_idx * k + nn];
                knn_dist_local_mem[local_idx][nn] = knn_dist[global_idx * k + nn];
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

                        if (is_candidate(knn_blocked[block])) {
#if defined(SYCL_LSH_NEAREST_NEIGHBOR_SEARCH_DISTRIBUTION_DEBUG)
                            // increment the counter
                            ++knn_calculation_count[hash_table * attr.rank_size + global_idx];
#endif

                            // calculate distances
                            for (index_type dim = 0; dim < attr.dims; ++dim) {
                                const real_type x = data_received[dim * attr.rank_size + global_idx];
                                const real_type y = data_owned[(knn_blocked[block] - base_id) * attr.dims + dim];
                                knn_dist_blocked[block] += (x - y) * (x - y);
                            }

                            // update nearest-neighbors
                            if (knn_dist_blocked[block] < knn_dist_local_mem[local_idx][0]) {
                                knn_local_mem[local_idx][0] = knn_blocked[block];
                                knn_dist_local_mem[local_idx][0] = knn_dist_blocked[block];

                                // ensure that the greatest distance is at pos 0 (bubble-sort)
                                for (index_type nn = 0; nn < k - 1; ++nn) {
                                    if (knn_dist_local_mem[local_idx][nn] < knn_dist_local_mem[local_idx][nn + 1]) {
                                        std::swap(knn_local_mem[local_idx][nn], knn_local_mem[local_idx][nn + 1]);
                                        std::swap(knn_dist_local_mem[local_idx][nn], knn_dist_local_mem[local_idx][nn + 1]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // write back to global buffer
            for (index_type nn = 0; nn < k; ++nn) {
                knn[global_idx * k + nn] = knn_local_mem[local_idx][nn];
                knn_dist[global_idx * k + nn] = knn_dist_local_mem[local_idx][nn];
            }
        });
    });

    // wait until all k-nearest-neighbors were calculated on the current MPI rank
    queue_.wait_and_throw();

    // add event if available
    if (profiler_ != nullptr) {
        profiler_->add_event(fmt::format("search_nearest_neighbors_round_{}_end", round));
    }

    return mpi_timer.elapsed();
}

}  // namespace sycl_lsh::detail::hashing

#endif  // SYCL_LSH_DETAIL_HASHING_HASH_TABLES_HPP
