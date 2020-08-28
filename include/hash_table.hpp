/**
 * @file
 * @author Marcel Breyer
 * @date 2020-08-28
 *
 * @brief Implements the @ref hash_tables class representing the used LSH hash tables.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP

#include <CL/sycl.hpp>
#include <iostream>

#include <config.hpp>
#include <data.hpp>
#include <detail/print.hpp>
#include <hash_functions/hash_function.hpp>
#include <knn.hpp>
#include <options.hpp>


/**
 * @brief Class representing the hash tables used in the LSH algorithm.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 * @tparam Data represents the used data
 */
template <memory_layout layout, template<memory_layout, typename, typename> typename HashFunctions, typename Options, typename Data>
class hash_tables : detail::hash_tables_base {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");
    static_assert(std::is_base_of_v<detail::data_base, Data>, "The third template parameter must by a 'data' type!");
public:
    /// The used floating point type.
    using real_type = typename Options::real_type;
    /// The used integer type.
    using index_type = typename Options::index_type;
    /// The used type of a hash value.
    using hash_value_type = typename Options::hash_value_type;


    /// The SYCL buffer holding all hash tables: `buffer.get_count() == options::num_hash_tables * data::size`.
    sycl::buffer<index_type , 1> buffer;
    /// The SYCL buffer holding the hash bucket offsets: `offsets.get_count() == options::num_hash_tables * (options::hash_table_size + 1)`.
    sycl::buffer<index_type, 1> offsets;
    /// Hash functions used by this hash tables.
    HashFunctions<layout, Options, Data> hash_function;

    template <typename Knns>
    void calculate_knn(const index_type k, Knns& knns) {
        static_assert(std::is_base_of_v<detail::knn_base, Knns>, "The template parameter must by a 'knn' type!");

        calculate_knn(k, data_.buffer, knns, true);
    }
    template <typename Knns>
    void calculate_knn(const index_type k, mpi_buffers<real_type, index_type>& data_mpi_buffers, Knns& knns) {
        static_assert(std::is_base_of_v<detail::knn_base, Knns>, "The template parameter must by a 'knn' type!");

        START_TIMING(copy_data_to_device);
        std::vector<real_type>& active_data_mpi_buffer = data_mpi_buffers.active();
        sycl::buffer<real_type, 1> data_buffer(active_data_mpi_buffer.size());
        auto acc = data_buffer.template get_access<sycl::access::mode::discard_write>();
        for (std::size_t i = 0; i < active_data_mpi_buffer.size(); ++i) {
            acc[i] = active_data_mpi_buffer[i];
        }
        END_TIMING_MPI(copy_data_to_device, comm_rank_);
        calculate_knn(k, data_buffer, knns);
    }
    template <typename Knns>
    void calculate_knn(const index_type k, sycl::buffer<real_type, 1>& data_buffer, Knns& knns, const bool first_round = false) {
        static_assert(std::is_base_of_v<detail::knn_base, Knns>, "The template parameter must by a 'knn' type!");

        // TODO 2020-06-23 18:54 marcel: implement correctly
        if (k > data_.rank_size) {
            throw std::invalid_argument("k must not be greater than the data set size!");
        }

        const index_type base_id = data_.total_size / comm_size_ * comm_rank_ + std::min<index_type>(comm_rank_, data_.total_size % comm_size_);

        START_TIMING(calculate_nearest_neighbors);
        queue_.submit([&](sycl::handler& cgh) {
            sycl::buffer<index_type, 1> knn_buffers(knns.buffers_knn.active().data(), knns.buffers_knn.active().size());
            sycl::buffer<real_type, 1> knn_buffers_dist(knns.buffers_dist.active().data(), knns.buffers_dist.active().size());

            auto acc_data_owned = data_.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_data_received = data_buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_hash_functions = hash_function.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_offsets = offsets.template get_access<sycl::access::mode::read>(cgh);
            auto acc_hash_tables = buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_knns = knn_buffers.template get_access<sycl::access::mode::write>(cgh);
            auto acc_knns_dist = knn_buffers_dist.template get_access<sycl::access::mode::read_write>(cgh);
            auto data = data_;
            auto opt = opt_;

            cgh.parallel_for<class kernel_calculate_knn>(sycl::range<>(data_.rank_size), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                if (idx >= data.rank_size) return;

                index_type* nearest_neighbors = new index_type[k];
                real_type* distances = new real_type[k];

                // initialize arrays
                for (index_type i = 0; i < k; ++i) {
                    nearest_neighbors[i] = acc_knns[Knns::get_linear_id(comm_rank_, idx, i, data, k)];
                    distances[i] = acc_knns_dist[Knns::get_linear_id(comm_rank_, idx, i, data, k)];
                }

                index_type argmax = 0;
                real_type max_distance = distances[argmax];
                for (index_type i = 0; i < k; ++i) {
                    if (distances[i] > max_distance) {
                        max_distance = distances[i];
                        argmax = i;
                    }
                }

                for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
                    const hash_value_type hash_bucket = hash_function.hash(comm_rank_, hash_table, idx, acc_data_received, acc_hash_functions, opt, data);

                    for (index_type bucket_element = acc_offsets[hash_table * (opt.hash_table_size + 1) + hash_bucket];
                            bucket_element < acc_offsets[hash_table * (opt.hash_table_size + 1) + hash_bucket + 1];
                            ++bucket_element)
                    {
                        const index_type point = acc_hash_tables[hash_table * data.rank_size + bucket_element];
                        const index_type point_idx = point - base_id;
                        real_type dist = 0.0;
                        for (index_type dim = 0; dim < data.dims; ++dim) {
                            const index_type x_idx = data.get_linear_id(comm_rank_, idx, data.rank_size, dim, data.dims);
                            const real_type x = acc_data_received[x_idx];
                            const index_type y_idx = data.get_linear_id(comm_rank_, point_idx, data.rank_size, dim, data.dims);
                            const real_type y = acc_data_owned[y_idx];

                            dist += (x - y) * (x - y);
                        }

                        // updated nearest-neighbors
                        const auto is_candidate = [=](const auto point, const index_type* neighbors, const index_type k, const index_type idx) {
                            if (first_round && point_idx == idx) return false;
                            for (index_type i = 0; i < k; ++i) {
                                if (neighbors[i] == point) return false;
                            }
                            return true;
                        };
                        if (dist < max_distance && is_candidate(point, nearest_neighbors, k, idx)) {
                            nearest_neighbors[argmax] = point;
                            distances[argmax] = dist;
                            max_distance = dist;
                            for (index_type i = 0; i < k; ++i) {
                                if (distances[i] > max_distance) {
                                    max_distance = distances[i];
                                    argmax = i;
                                }
                            }
                        }
                    }
                }

                // write back to result buffer
                for (index_type i = 0; i < k; ++i) {
                    acc_knns[Knns::get_linear_id(comm_rank_, idx, i, data, k)] = nearest_neighbors[i];
                    acc_knns_dist[Knns::get_linear_id(comm_rank_, idx, i, data, k)] = distances[i];
                }

                delete[] nearest_neighbors;
                delete[] distances;
            });
        });
        END_TIMING_MPI_AND_BARRIER(calculate_nearest_neighbors, comm_rank_, queue_);
    }


    [[nodiscard]] constexpr index_type get_linear_id(const index_type hash_table, const hash_value_type hash_value) const noexcept {
        // TODO 2020-05-11 17:17 marcel: implement correctly
        return hash_table * data_.rank_size + static_cast<index_type>(hash_value);
    }

    /**
     * @brief Returns the @ref options object which has been used to create this @ref hash_tables object.
     * @return the @ref options object (`[[nodiscard]]`)
     */
    [[nodiscard]] const Options& get_options() const noexcept { return opt_; }
    /**
     * @brief Returns the @ref data object which has been used to create this @ref hash_tables object.
     * @return the @ref data object (`[[nodiscard]]`)
     */
    [[nodiscard]] Data& get_data() const noexcept { return data_; }
    /**
     * @brief Returns the specified @ref memory_layout (*Array of Structs* or *Struct of Arrays*).
     * @return the specified @ref memory_layout (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr memory_layout get_memory_layout() const noexcept { return layout; }

private:
    /// Befriend factory function.
    template <memory_layout layout_, template<memory_layout, typename, typename> typename HashFunctions_, typename Options_, typename Data_>
    friend auto make_hash_tables(sycl::queue&, HashFunctions_<layout_, Options_, Data_>, const MPI_Comm&);


    /**
     * @brief Construct new hash tables given the options in @p opt, the sizes in @p data and the hash functions in @p hash_functions.
     * @param[inout] queue the SYCL command queue
     * @param[in] opt the @ref options object representing the currently set options
     * @param[in] data the @ref data object representing the used data set
     * @param[in] hash_functions the hash functions object representing the used LSH hash functions
     * @param[in] comm_rank the current MPI rank
     * @param[in] comm_size the current MPI comm size
     */
    hash_tables(sycl::queue& queue, const Options& opt, Data& data, HashFunctions<layout, Options, Data> hash_functions, const int comm_rank, const int comm_size)
            : buffer(opt.num_hash_tables * data.rank_size), offsets(opt.num_hash_tables * (opt.hash_table_size + 1)),
              hash_function(hash_functions), queue_(queue), comm_rank_(comm_rank), comm_size_(comm_size), opt_(opt), data_(data)
    {
        {
            // create temporary buffer to count the occurrence of each hash value
            sycl::buffer<index_type, 1> hash_value_count(opt_.num_hash_tables * opt_.hash_table_size);

            // TODO 2020-05-11 17:28 marcel: implement optimizations
            // count the occurrence of each hash value
            this->count_hash_values(hash_value_count);

            // write distribution to file
//            std::ofstream out("../evaluation/entropy_bucket_distribution_2.txt");
//            auto acc = hash_value_count.template get_access<sycl::access::mode::read>();
//            for (index_type hash_table = 0; hash_table < opt_.num_hash_tables; ++hash_table) {
//                for (index_type i = 0; i < opt_.hash_table_size - 1; ++i) {
//                    out << acc[hash_table * opt_.hash_table_size + i] << ',';
//                }
//                out << acc[hash_table * opt_.hash_table_size + opt_.hash_table_size - 1] << std::endl;
//            }

            // calculate the offset values
            this->calculate_offsets(hash_value_count);
        }
        // fill the hash tables based on the previously calculated offset values
        this->fill_hash_tables();
    }


    /**
     * @brief Calculates the number of data points assigned to each hash bucket in each hash table.
     * @param[inout] hash_value_count the number of data points assigned to each hash bucket in each hash table
     */
    void count_hash_values(sycl::buffer<index_type, 1>& hash_value_count) {
        START_TIMING(count_hash_values);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_hash_value_count = hash_value_count.template get_access<sycl::access::mode::atomic>(cgh);
            auto acc_hash_functions = hash_function.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_data = data_.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto opt = opt_;
            auto data = data_;
            auto comm_rank = comm_rank_;

            cgh.parallel_for<class kernel_count_hash_values>(sycl::range<>(data.rank_size), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                if (idx >= data.rank_size) return;

                for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
                    const hash_value_type hash_value =
                            hash_function.hash(comm_rank, hash_table, idx, acc_data, acc_hash_functions, opt, data);
                    acc_hash_value_count[hash_table * opt.hash_table_size + hash_value].fetch_add(1);
                }
            });
        });
        END_TIMING_MPI_AND_BARRIER(count_hash_values, comm_rank_, queue_);
    }
    /**
     * @brief Calculates the offsets for each hash bucket in each hash table.
     * @param[in] hash_value_count the number of data points assigned to each hash bucket in each hash table.
     */
    void calculate_offsets(sycl::buffer<index_type, 1>& hash_value_count) {
        START_TIMING(calculate_offsets);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_hash_value_count = hash_value_count.template get_access<sycl::access::mode::read>(cgh);
            auto acc_offsets = offsets.template get_access<sycl::access::mode::discard_write>(cgh);
            auto opt = opt_;

            cgh.parallel_for<class kernel_calculate_offsets>(sycl::range<>(opt.num_hash_tables), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                // calculate constant offsets
                const index_type hash_table_offset = idx * (opt.hash_table_size + 1);
                const index_type hash_value_count_offset = idx * opt.hash_table_size;
                // zero out first two offsets in each hash table
                acc_offsets[hash_table_offset] = 0;
                acc_offsets[hash_table_offset + 1] = 0;
                for (index_type hash_value = 2; hash_value <= opt.hash_table_size; ++hash_value) {
                    // calculate modified prefix sum
                    acc_offsets[hash_table_offset + hash_value] =
                            acc_offsets[hash_table_offset + hash_value - 1] +
                            acc_hash_value_count[hash_value_count_offset + hash_value - 2];
                }
            });
        });
        END_TIMING_MPI_AND_BARRIER(calculate_offsets, comm_rank_, queue_);
    }
    /**
     * @brief Fill the hash tables with the data points using the previously calculated offsets.
     */
    void fill_hash_tables() {
        START_TIMING(fill_hash_tables);
        const index_type base_id = data_.total_size / comm_size_ * comm_rank_ + std::min<index_type>(comm_rank_, data_.total_size % comm_size_);
        const bool has_smaller_rank_size = (data_.total_size % comm_size_ != 0) && static_cast<index_type>(comm_rank_) >= (data_.total_size % comm_size_);
        
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_data = data_.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_hash_functions = hash_function.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_offsets = offsets.template get_access<sycl::access::mode::atomic>(cgh);
            auto acc_hash_tables = buffer.template get_access<sycl::access::mode::discard_write>(cgh);
            auto opt = opt_;
            auto data = data_;
            auto comm_rank = comm_rank_;

            cgh.parallel_for<class kernel_fill_hash_tables>(sycl::range<>(data.rank_size), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                index_type val = 0;
                if (has_smaller_rank_size && idx == data.rank_size - 1) {
                    val = base_id;
                } else {
                    val = idx + base_id;
                }

                for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
                    const hash_value_type hash_value = hash_function.hash(comm_rank, hash_table, idx, acc_data, acc_hash_functions, opt, data);
                    acc_hash_tables[hash_table * data.rank_size + acc_offsets[hash_table * (opt.hash_table_size + 1) + hash_value + 1].fetch_add(1)]
                        = val;
                }
            });
        });
        END_TIMING_MPI_AND_BARRIER(fill_hash_tables, comm_rank_, queue_);
    }

    /// Reference to the SYCL queue object.
    sycl::queue queue_;
    /// The current MPI rank.
    const int comm_rank_;
    /// The current MPI comm size.
    const int comm_size_;
    /// Const reference to @ref options object.
    const Options& opt_;
    /// Reference to @ref data object.
    Data& data_;
};


/**
 * @brief Factory function for creating a new @ref hash_tables object.
 * @tparam layout the @ref memory_layout type
 * @tparam Options the @ref options type
 * @tparam Data the @ref data type
 * @param[inout] queue the SYCL command queue
 * @param[in] hash_functions the used hash functions for this hash tables
 * @param[in] communicator the *MPI_Comm* communicator
 * @return the newly constructed @ref hash_tables object (`[[nodiscard]]`)
 */
template <memory_layout layout, template<memory_layout, typename, typename> typename HashFunctions, typename Options, typename Data>
[[nodiscard]] inline auto make_hash_tables(sycl::queue& queue, HashFunctions<layout, Options, Data> hash_functions, const MPI_Comm& communicator) {
    int comm_rank, comm_size;
    MPI_Comm_rank(communicator, &comm_rank);
    MPI_Comm_size(communicator, &comm_size);
    return hash_tables<layout, HashFunctions, Options, Data>(queue, hash_functions.get_options(), hash_functions.get_data(), hash_functions, comm_rank, comm_size);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP