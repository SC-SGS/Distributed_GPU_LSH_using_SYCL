#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP

#include <config.hpp>
#include <options.hpp>
#include <data.hpp>

#include <CL/sycl.hpp>
#include <random>
#include <iostream>


template <typename real_type, typename size_type, typename hash_value_type>
class hash_tables {
public:

    template <typename Data, typename Options>
    hash_tables(sycl::queue& queue, Data& data, const Options& opt)
        : hash_functions_(sycl::range<1>{ opt.num_hash_tables * opt.num_hash_functions * (data.dims + 1) }),
          hash_tables_(sycl::range<1>{ opt.num_hash_tables * data.size })
    {
        this->init_hash_functions(data, opt);

        std::vector<size_type> hash_values_count(opt.num_hash_tables * opt.hash_table_size, 0.0);
        sycl::buffer buf_hash_values_count(hash_values_count.data(), sycl::range<1>{ hash_values_count.size() });

        // TODO 2020-04-29 16:31 marcel:
        const auto previous_power_of_two = [&] (const auto local_mem_size) -> std::size_t {
            return std::pow(2, std::floor(std::log2(local_mem_size / (data.dims * sizeof(real_type)))));
        };

        const size_type local_mem_size = queue.get_device().get_info<sycl::info::device::local_mem_size>();
        const auto max_work_item_sizes = queue.get_device().get_info<sycl::info::device::max_work_item_sizes>();
        const size_type local_size = std::min(previous_power_of_two(local_mem_size), max_work_item_sizes[0]);
        const auto round_up = [](const auto x, const auto y) { return ((x + y - 1) / y) * y; };
        const size_type global_size = round_up(data.size, local_size);

        queue.submit([&](sycl::handler& cgh) {
            auto acc_count = buf_hash_values_count.template get_access<sycl::access::mode::atomic>(cgh);
            auto acc_hash_func = hash_functions_.template get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);
            auto acc_data = data.data_.template get_access<sycl::access::mode::read>(cgh);

            sycl::accessor<real_type, 1, sycl::access::mode::read_write, sycl::access::target::local>
                    local_mem(sycl::range<1>{ local_size * data.dims }, cgh);

            auto execution_range = sycl::nd_range<1>{ sycl::range<1>{ global_size }, sycl::range<1>{ local_size } };

            cgh.parallel_for<class count_hash_values>(execution_range, [=](sycl::nd_item<1> item) {
                const size_type idx = item.get_global_linear_id();
                const size_type local_idx = item.get_local_linear_id();

                if (idx >= data.size) { return; }

                // copy to local memory
                for (size_type i = 0; i < data.dims; ++i) {
//                    local_mem[local_idx + i * local_size] = acc_data[i + idx * data.dims]; // SoA
                    local_mem[i + local_idx * data.dims] = acc_data[i + idx * data.dims]; // AoS
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (size_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
//                    hash += local_mem[d + local_idx * opt.dims] * acc_hash_func[d + j * (opt.dims + 1)]; // AoS
                    const size_type hash_value = this->hash(local_idx, acc_data, acc_hash_func, data, opt);
                    acc_count[hash_value + hash_table * opt.hash_table_size].fetch_add(1);
                }
            });
        });
    }

    template <memory_type layout, typename Data, typename Options>
    size_type get_linear_idx(const size_type hash_table, const size_type hash_function, const size_type dim,
                    const Data& data, const Options& opt) noexcept {
        if constexpr (layout == memory_type::aos) {
            // grouped by hash_tables -> grouped by hash_functions
            return hash_table * (opt.num_hash_functions * (data.dims + 1))
                   + hash_function * (data.dims + 1)
                   + dim;
        } else {
            // grouped by hash_tables -> grouped by dimensions
            return hash_table * (opt.num_hash_functions * (data.dims + 1))
                   + dim * opt.num_hash_functions
                   + hash_function;
        }
    }

//private:
    sycl::buffer<real_type, 1> hash_functions_;
    sycl::buffer<real_type, 1> hash_tables_;

private:

    template <typename Data, typename Options>
    void init_hash_functions(const Data& data, const Options& opt) {
//    std::random_device rnd_device;
//    std::mt19937 rnd_normal_gen(rnd_device());
//    std::mt19937 rnd_uniform_gen(rnd_device());
        std::mt19937 rnd_normal_gen;
        std::mt19937 rnd_uniform_gen;
        std::normal_distribution<real_type> rnd_normal_dist;
        std::uniform_real_distribution<real_type> rnd_uniform_dist(0, opt.w);

        auto acc = hash_functions_.template get_access<sycl::access::mode::discard_write>();
        for (size_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
            for (size_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
                for (size_type dim = 0; dim < data.dims; ++dim) {
                    acc[this->get_linear_idx<layout>(hash_table, hash_function, dim, data, opt)] = std::abs(rnd_uniform_dist(rnd_normal_gen));
                }
                acc[this->get_linear_idx<layout>(hash_table, hash_function, data.dims, data, opt)] = rnd_uniform_dist(rnd_uniform_gen);
            }
        }
    }

    template <typename AccData, typename AccHashFunction, typename Data, typename Options>
    hash_value_type hash(const size_type idx_data, AccData& acc_data, AccHashFunction& acc_hash_func,
            const Data& data, const Options& opt)
    {
        hash_value_type combined_hash = opt.num_hash_functions;
        for (size_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            real_type hash = acc_hash_func[data.dims + hash_function * (data.dims + 1)];
            for (size_type dim = 0; dim < data.dims; ++dim) {
//                hash += local_mem[local_idx + d * local_size] * acc_hash_func[d + j * (opt.dims + 1)]; // SoA
                hash += acc_data[dim + idx_data * data.dims] * acc_hash_func[dim + hash_function * (data.dims + 1)]; // Aos
//                hash += acc_data[idx_data] * acc_hash_func[idx_hash_func]; // Aos
            }
            combined_hash ^= static_cast<hash_value_type>(hash / opt.w) + 0x9e3779b9 + (combined_hash << 6u) + (combined_hash >> 2u);
        }
        if constexpr (!std::is_signed_v<hash_value_type>) {
            if (combined_hash < 0) {
                combined_hash *= -1;
            }
        }
        return combined_hash %= opt.hash_table_size;
    }
};


template <typename Data, typename Options>
hash_tables(sycl::queue&, const Data&, const Options&)
    -> hash_tables<typename Options::real_type, typename Options::size_type, typename Options::hash_value_type>;



#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_TABLE_HPP
