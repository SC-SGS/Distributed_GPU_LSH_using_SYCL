/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-18
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
#include <hash_function.hpp>
#include <knn.hpp>
#include <options.hpp>


template <typename hash_value_type, typename index_type, typename real_type, typename AccData, typename AccHashFunction>
[[nodiscard]] hash_value_type hash(const index_type hash_table, const index_type point,
                                   AccData& acc_data, AccHashFunction& acc_hash_function,
                                   index_type num_hash_functions, real_type w, index_type hash_table_size, index_type dims)
{
    hash_value_type combined_hash = num_hash_functions;
    for (index_type hash_function = 0; hash_function < num_hash_functions; ++hash_function) {
        real_type hash = acc_hash_function[hash_table * num_hash_functions * (dims + 1) + hash_function * (dims + 1) + dims];
        for (index_type dim = 0; dim < dims; ++dim) {
            hash += acc_data[dim + point * dims] * acc_hash_function[hash_table * num_hash_functions * (dims + 1) + hash_function * (dims + 1) + dim];
        }
        combined_hash ^= static_cast<hash_value_type>(hash / w)
                         + static_cast<hash_value_type>(0x9e3779b9)
                         + (combined_hash << static_cast<hash_value_type>(6))
                         + (combined_hash >> static_cast<hash_value_type>(2));
    }
    if constexpr (std::is_signed_v<hash_value_type>) {
        combined_hash = combined_hash < 0 ? -combined_hash : combined_hash;
    }
    return combined_hash %= hash_table_size;
}


namespace detail {
    /**
     * @brief Empty base class for the @ref hash_tables class. Only for static_asserts.
     */
    class hash_tables_base {};
}


/**
 * @brief Class representing the hash tables used in the LSH algorithm.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 * @tparam Data represents the used data
 */
template <memory_layout layout, typename Options, typename Data>
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
    hash_functions<layout, Options, Data> hash_function;


    template <typename Knns>
    void calculate_knn(const index_type k, mpi_buffers<real_type, index_type>& data_mpi_buffers, Knns& knns) {
        // TODO 2020-06-23 18:54 marcel: implement correctly
        if (k > data_.size) {
            throw std::invalid_argument("k must not be greater than data.size!");
        }

        START_TIMING(calculate_nearest_neighbors);

        queue_.submit([&](sycl::handler& cgh) {
            sycl::buffer<index_type, 1> knn_buffers(knns.buffers.active(), knns.buffers.size * knns.buffers.dims);

            auto acc_data = data_.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_hash_functions = hash_function.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_offsets = offsets.template get_access<sycl::access::mode::read>(cgh);
            auto acc_hash_tables = buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_knns = knn_buffers.template get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class kernel_calculate_knn>(sycl::range<>(data_.size), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                if (idx >= data_.size) return;

                index_type* nearest_neighbors = new index_type[k];
                real_type* distances = new real_type[k];
                real_type max_distance = std::numeric_limits<real_type>::max();
                index_type argmax = 0;

                // initialize arrays
                for (index_type i = 0; i < k; ++i) {
                    nearest_neighbors[i] = idx;
                    distances[i] = max_distance;
                }

                for (index_type hash_table = 0; hash_table < opt_.num_hash_tables; ++hash_table) {
                    const hash_value_type hash_bucket = hash_function.hash(hash_table, idx, acc_data, acc_hash_functions);

                    for (index_type bucket_element = acc_offsets[hash_table * (opt_.hash_table_size) + hash_bucket];
                            bucket_element < acc_offsets[hash_table * (opt_.hash_table_size) + hash_bucket + 1];
                            ++bucket_element)
                    {
                        const index_type point = acc_hash_tables[hash_table * data_.size + bucket_element];
                        real_type dist = 0.0;
                        for (index_type dim = 0; dim < data_.dims; ++dim) {
                            dist += (acc_data[data_.get_linear_id(idx, dim)] - acc_data[data_.get_linear_id(point, dim)])
                                    * (acc_data[data_.get_linear_id(idx, dim)] - acc_data[data_.get_linear_id(point, dim)]);
                        }

                        // updated nearest-neighbors
                        auto contains = [](const auto point, const index_type* neighbors, const index_type k) {
                            for (index_type i = 0; i < k; ++i) {
                                if (neighbors[i] == point) return true;
                            }
                            return false;
                        };
                        if (dist < max_distance && !contains(point, nearest_neighbors, k)) {
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
                    acc_knns[Knns::get_linear_id(idx, data_.size, i, k)] = nearest_neighbors[i];
                }

                delete[] nearest_neighbors;
                delete[] distances;
            });
        });

        END_TIMING_MPI_AND_BARRIER(calculate_nearest_neighbors, comm_rank_, queue_);
    }


    [[nodiscard]] constexpr index_type get_linear_id(const index_type hash_table, const hash_value_type hash_value) const noexcept {
        // TODO 2020-05-11 17:17 marcel: implement correctly
        return hash_table * data_.size + static_cast<index_type>(hash_value);
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
    template <memory_layout layout_, typename Options_, typename Data_>
    friend auto make_hash_tables(sycl::queue&, hash_functions<layout_, Options_, Data_>, const MPI_Comm&);


    /**
     * @brief Construct new hash tables given the options in @p opt, the sizes in @p data and the hash functions in @p hash_functions.
     * @param[inout] queue the SYCL command queue
     * @param[in] opt the @ref options object representing the currently set options
     * @param[in] data the @ref data object representing the used data set
     * @param[in] hash_functions the @ref hash_functions object representing the used LSH hash functions
     * @param[in] comm_rank the current MPI rank
     */
    hash_tables(sycl::queue& queue, const Options& opt, Data& data, hash_functions<layout, Options, Data> hash_functions, const int comm_rank)
            : buffer(opt.num_hash_tables * data.size), offsets(opt.num_hash_tables * (opt.hash_table_size + 1)),
              hash_function(hash_functions), queue_(queue), comm_rank_(comm_rank), opt_(opt), data_(data)
    {
        {
            // create temporary buffer to count the occurrence of each hash value
            std::vector<index_type> vec(opt_.num_hash_tables * opt_.hash_table_size, index_type{0});
            sycl::buffer hash_value_count(vec.data(), sycl::range<>(vec.size()));

            // TODO 2020-05-11 17:28 marcel: implement optimizations
            // count the occurrence of each hash value
            this->count_hash_values(hash_value_count);

            // calculate the offset values
//            this->calculate_offsets(hash_value_count);
            queue_.wait_and_throw();
        }
        // fill the hash tables based on the previously calculated offset values
//        this->fill_hash_tables();
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
            const index_type data_size = data_.size;
            const index_type opt_num_hash_tables = opt_.num_hash_tables;
            const index_type opt_hash_table_size = opt_.hash_table_size;
            const index_type opt_num_hash_functions = opt_.num_hash_functions;
            const index_type opt_w = opt_.w;
            const index_type data_dims = data_.dims;

            cgh.parallel_for<class kernel_count_hash_values>(sycl::range<>(data_size), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                if (idx >= data_size) return;

                for (index_type hash_table = 0; hash_table < opt_num_hash_tables; ++hash_table) {
                    const hash_value_type hash_value =
                            hash<hash_value_type, index_type, real_type>(hash_table, idx, acc_data, acc_hash_functions, opt_num_hash_functions, opt_w, opt_hash_table_size, data_dims);
                    acc_hash_value_count[hash_table * opt_hash_table_size + hash_value].fetch_add(1);
                }
            });
        });
        END_TIMING_MPI(count_hash_values, comm_rank_);
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

            cgh.parallel_for<class kernel_calculate_offsets>(sycl::range<>(opt_.num_hash_tables), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                // calculate constant offsets
                const index_type hash_table_offset = idx * (opt_.hash_table_size + 1);
                const index_type hash_value_count_offset = idx * opt_.hash_table_size;
                // zero out first two offsets in each hash table
                acc_offsets[hash_table_offset] = 0;
                acc_offsets[hash_table_offset + 1] = 0;
                for (index_type hash_value = 2; hash_value <= opt_.hash_table_size; ++hash_value) {
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
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_data = data_.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_hash_functions = hash_function.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_offsets = offsets.template get_access<sycl::access::mode::atomic>(cgh);
            auto acc_hash_tables = buffer.template get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class kernel_fill_hash_tables>(sycl::range<>(data_.size), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                for (index_type hash_table = 0; hash_table < opt_.num_hash_tables; ++hash_table) {
                    const hash_value_type hash_value = hash_function.hash(hash_table, idx, acc_data, acc_hash_functions);
                    acc_hash_tables[hash_table * data_.size + acc_offsets[hash_table * (opt_.hash_table_size + 1) + hash_value + 1].fetch_add(1)] = idx;
                }
            });
        });
        END_TIMING_MPI_AND_BARRIER(fill_hash_tables, comm_rank_, queue_);
    }

    /// Reference to the SYCL queue object.
    sycl::queue queue_;
    /// The current MPI rank.
    const int comm_rank_;
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
 * @param[in] hash_functions the used @ref hash_functions for this hash tables
 * @param[in] communicator the *MPI_Comm* communicator
 * @return the newly constructed @ref hash_tables object (`[[nodiscard]]`)
 */
template <memory_layout layout, typename Options, typename Data>
[[nodiscard]] inline auto make_hash_tables(sycl::queue& queue, hash_functions<layout, Options, Data> hash_functions, const MPI_Comm& communicator) {
    int comm_rank;
    MPI_Comm_rank(communicator, &comm_rank);
    return hash_tables<layout, Options, Data>(queue, hash_functions.get_options(), hash_functions.get_data(), hash_functions, comm_rank);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP
