/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-12
 *
 * @brief Implements the @ref hash_tables class representing the used LSH hash tables.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP

#include <config.hpp>
#include <options.hpp>
#include <data.hpp>
#include <detail/print.hpp>
#include <hash_function.hpp>

#include <CL/sycl.hpp>
#include <random>
#include <iostream>


/**
 * @brief
 * @tparam layout
 * @tparam Options
 * @tparam Data
 */
template <memory_layout layout, typename Options, typename Data>
class hash_tables {
public:
    using real_type = typename Options::real_type;
    using index_type = typename Options::index_type;
    using hash_value_type = typename Options::hash_value_type;




    [[nodiscard]] constexpr index_type get_linear_idx(const index_type hash_table, const hash_value_type hash_value) const noexcept {
        // TODO 2020-05-11 17:17 marcel: implement correctly
        return hash_table * data_.size + static_cast<index_type>(hash_value);
    }
    [[nodiscard]] constexpr memory_layout get_memory_layout() const noexcept {
        return layout;
    }


    sycl::buffer<real_type, 1> buffer;
    sycl::buffer<index_type, 1> offsets;

private:
    /// Befriend factory function.
    template <memory_layout layout_, typename Data_>
    friend hash_tables<layout_, typename Data_::options_type, Data_> make_hash_tables(sycl::queue&, Data_&);
    /// Befriend factory function.
    template <memory_layout layout_, typename Options_, typename Data_>
    friend hash_tables<layout_, Options_, Data_> make_hash_tables(sycl::queue&, hash_functions<layout_, Options_, Data_>);


    hash_tables(sycl::queue& queue, const Options& opt, Data& data, hash_functions<layout, Options, Data> hash_functions)
            : opt_(opt), data_(data), hash_functions_(hash_functions),
              buffer(opt.num_hash_tables * data.size), offsets(opt.num_hash_tables * (opt.hash_table_size + 1))
    {
        // create temporary buffer to count the occurrence of each hash value
        std::vector<index_type> vec(opt_.num_hash_tables * opt_.hash_table_size, index_type{0});
        sycl::buffer hash_value_count(vec.data(), sycl::range<>(vec.size()));

        // TODO 2020-05-11 17:28 marcel: implement optimizations
        // count the occurrence of each hash value
        this->count_hash_values(queue, hash_value_count);
        // calculate the offset values
        this->calculate_offsets(queue, hash_value_count);
        // fill the hash tables based on the previously calculated offset values
        this->fill_hash_tables(queue);
    }


    hash_tables(sycl::queue& queue, const Options& opt, Data& data)
            : hash_tables(queue, opt, data, make_hash_functions<layout>(data)) { }


    void count_hash_values(sycl::queue& queue, sycl::buffer<index_type, 1>& hash_value_count) {
        queue.submit([&](sycl::handler& cgh) {
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
    }
    void calculate_offsets(sycl::queue& queue, sycl::buffer<index_type, 1>& hash_value_count) {
        queue.submit([&](sycl::handler& cgh) {
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
    }
    void fill_hash_tables(sycl::queue& queue) {
        queue.submit([&](sycl::handler& cgh) {
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
    }


    /// Const reference to @ref options object.
    const Options& opt_;
    /// Reference to @ref data object.
    Data& data_;
    /// Hash functions used by this hash tables.
    hash_functions<layout, Options, Data> hash_functions_;

};


template <memory_layout layout, typename Data>
[[nodiscard]] inline hash_tables<layout, typename Data::options_type, Data> make_hash_tables(sycl::queue& queue, Data& data) {
    return hash_tables<layout, typename Data::options_type, Data>(queue, data.get_options(), data);
}

template <memory_layout layout, typename Options, typename Data>
[[nodiscard]] inline hash_tables<layout, Options, Data> make_hash_tables(sycl::queue& queue, hash_functions<layout, Options, Data> hash_func)
{
    return hash_tables<layout, Options, Data>(queue, hash_func.get_options(), hash_func.get_data(), hash_func);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP
