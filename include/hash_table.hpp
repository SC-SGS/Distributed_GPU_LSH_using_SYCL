/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-15
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


/**
 * @brief Class representing the hash tables used in the LSH algorithm.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 * @tparam Data represents the used data
 */
template <memory_layout layout, typename Options, typename Data>
class hash_tables {
    /// The used floating point type.
    using real_type = typename Options::real_type;
    /// The used integer type.
    using index_type = typename Options::index_type;
    /// The used type of a hash value.
    using hash_value_type = typename Options::hash_value_type;
public:


    /// The SYCL buffer holding all hash hables: `buffer.get_count() == options::num_hash_tables * data::size`.
    sycl::buffer<real_type, 1> buffer;
    /// The SYCL buffer holding the hash bucket offsets: `offsets.get_count() == options::num_hash_tables * (options::hash_table_size + 1)`.
    sycl::buffer<index_type, 1> offsets;
    /// Hash functions used by this hash tables.
    hash_functions<layout, Options, Data> hash_functions_;


    template <memory_layout knn_layout>
    auto calculate_knn(const index_type k) {
        START_TIMING(calculate_nearest_neighbours);
        auto knns = make_knn<knn_layout>(k, data_);

        END_TIMING_WITH_BARRIER(calculate_nearest_neighbours, queue_);
        return knns;
    }


    [[nodiscard]] constexpr index_type get_linear_idx(const index_type hash_table, const hash_value_type hash_value) const noexcept {
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
    template <memory_layout layout_, typename Data_>
    friend hash_tables<layout_, typename Data_::options_type, Data_> make_hash_tables(sycl::queue&, Data_&);
    /// Befriend factory function.
    template <memory_layout layout_, typename Options_, typename Data_>
    friend hash_tables<layout_, Options_, Data_> make_hash_tables(sycl::queue&, hash_functions<layout_, Options_, Data_>);


    /**
     * @brief Construct new hash tables given the options in @p opt, the sizes in @p data and the hash functions in @p hash_functions.
     * @param[inout] queue the SYCL command queue
     * @param[in] opt the @ref options object representing the currently set options
     * @param[in] data the @ref data object representing the used data set
     * @param[in] hash_functions the @ref hash_functions object representing the used LSH hash functions
     */
    hash_tables(sycl::queue& queue, const Options& opt, Data& data, hash_functions<layout, Options, Data> hash_functions)
            : queue_(queue), opt_(opt), data_(data), hash_functions_(hash_functions),
              buffer(opt.num_hash_tables * data.size), offsets(opt.num_hash_tables * (opt.hash_table_size + 1))
    {
        {
            // create temporary buffer to count the occurrence of each hash value
            std::vector<index_type> vec(opt_.num_hash_tables * opt_.hash_table_size, index_type{0});
            sycl::buffer hash_value_count(vec.data(), sycl::range<>(vec.size()));

            // TODO 2020-05-11 17:28 marcel: implement optimizations
            // count the occurrence of each hash value
            this->count_hash_values(hash_value_count);

            // calculate the offset values
            this->calculate_offsets(hash_value_count);
        }
        // fill the hash tables based on the previously calculated offset values
        this->fill_hash_tables();
    }

    /**
     * @brief Construct new hash tables given the options in @p opt and sizes in @p data.
     * @details Internally constructs a @ref hash_functions object with the parameters givne in @p opt and @p data.
     * @param[inout] queue the SYCL command queue
     * @param[in] opt the @ref options object representing the currently set options
     * @param[in] data the @ref data object representing the used data set
     */
    hash_tables(sycl::queue& queue, const Options& opt, Data& data)
            : hash_tables(queue, opt, data, make_hash_functions<layout>(data)) { }


    /**
     * @brief Calculates the number of data points assigned to each hash bucket in each hash table.
     * @param[inout] hash_value_count the number of data points assigned to each hash bucket in each hash table
     */
    void count_hash_values(sycl::buffer<index_type, 1>& hash_value_count) {
        START_TIMING(count_hash_values);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_hash_value_count = hash_value_count.template get_access<sycl::access::mode::atomic>(cgh);
            auto acc_hash_functions = hash_functions_.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_data = data_.buffer.template get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for<class kernel_count_hash_values>(sycl::range<>(data_.size), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                if (idx >= data_.size) return;

                for (index_type hash_table = 0; hash_table < opt_.num_hash_tables; ++hash_table) {
                    const hash_value_type hash_value = hash_functions_.hash(hash_table, idx, acc_data, acc_hash_functions);
                    acc_hash_value_count[hash_table * opt_.hash_table_size + hash_value].fetch_add(1);
                }
            });
        });
        END_TIMING_WITH_BARRIER(count_hash_values, queue_);
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

                index_type offset_value = data_.size;
                acc_offsets[idx * (opt_.hash_table_size + 1)] = 0;

                for (index_type hash_value = opt_.hash_table_size; hash_value > 0; --hash_value) {
                    offset_value -= acc_hash_value_count[idx * opt_.hash_table_size + hash_value - 1];
                    acc_offsets[idx * (opt_.hash_table_size + 1) + hash_value] = offset_value;
                }
            });
        });
        END_TIMING_WITH_BARRIER(calculate_offsets, queue_);
    }
    /**
     * @brief Fill the hash tables with the data points using the previously calculated offsets.
     */
    void fill_hash_tables() {
        START_TIMING(fill_hash_tables);
        queue_.submit([&](sycl::handler& cgh) {
            auto acc_data = data_.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_hash_functions = hash_functions_.buffer.template get_access<sycl::access::mode::read>(cgh);
            auto acc_offsets = offsets.template get_access<sycl::access::mode::atomic>(cgh);
            auto acc_hash_tables = buffer.template get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class kernel_fill_hash_tables>(sycl::range<>(data_.size), [=](sycl::item<> item) {
                const index_type idx = item.get_linear_id();

                for (index_type hash_table = 0; hash_table < opt_.num_hash_tables; ++hash_table) {
                    const hash_value_type  hash_value = hash_functions_.hash(hash_table, idx, acc_data, acc_hash_functions);
                    acc_hash_tables[hash_table * data_.size + acc_offsets[hash_table * (opt_.hash_table_size + 1) + hash_value + 1].fetch_add(1)] = idx;
                }
            });
        });
        END_TIMING_WITH_BARRIER(fill_hash_tables, queue_);
    }

    /// Reference to the SYCL queue object.
    sycl::queue queue_;
    /// Const reference to @ref options object.
    const Options& opt_;
    /// Reference to @ref data object.
    Data& data_;
};


/**
 * @brief Factory function for creating a new @ref hash_tables object.
 * @tparam layout the @ref memory_layout type
 * @tparam Data the @ref data type
 * @param[inout] queue the SYCL command queue
 * @param[in] data the used data object
 * @return the newly constructed @ref hash_tables object (`[[nodiscard]]`)
 */
template <memory_layout layout, typename Data>
[[nodiscard]] inline hash_tables<layout, typename Data::options_type, Data> make_hash_tables(sycl::queue& queue, Data& data) {
    return hash_tables<layout, typename Data::options_type, Data>(queue, data.get_options(), data);
}

/**
 * @brief Factory function for creating a new @ref hash_tables object.
 * @tparam layout the @ref memory_layout type
 * @tparam Options the @ref options type
 * @tparam Data the @ref data type
 * @param[inout] queue the SYCL command queue
 * @param[in] hash_functions the used @ref hash_functions for this hash tables
 * @return the newly constructed @ref hash_tables object (`[[nodiscard]]`)
 */
template <memory_layout layout, typename Options, typename Data>
[[nodiscard]] inline hash_tables<layout, Options, Data> make_hash_tables(sycl::queue& queue, hash_functions<layout, Options, Data> hash_functions)
{
    return hash_tables<layout, Options, Data>(queue, hash_functions.get_options(), hash_functions.get_data(), hash_functions);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP
