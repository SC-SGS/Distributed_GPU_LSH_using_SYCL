#include <iostream>

#include <CL/sycl.hpp>
#include <random>
#include <cmath>
#include <cassert>
#include <type_traits>

#include <detail/print.hpp>
#include <hash_table.hpp>

namespace sycl = cl::sycl;


template <typename InputIt>
void print(InputIt first, InputIt last) {
    std::for_each(first, last, [](const auto val) { std::cout << val << " "; });
    std::cout << std::endl;
}

using real_t = float;
using index_t = uint32_t;

int main() {
    constexpr std::size_t dim = 3;
    constexpr std::size_t number_of_hashes = 3;
    // w
    constexpr real_t w = 1.0;
    constexpr index_t prim = 105613;
    constexpr std::size_t size = 2048;

    std::vector<real_t> hash_functions(number_of_hashes * (dim + 1));

//    std::random_device rnd_device;
//    std::mt19937 rnd_normal_gen(rnd_device());
//    std::mt19937 rnd_uniform_gen(rnd_device());
    std::mt19937 rnd_normal_gen;
    std::mt19937 rnd_uniform_gen;
    std::normal_distribution<real_t> rnd_normal_dist;
    std::uniform_real_distribution<real_t> rnd_uniform_dist(0, w);

    for (std::size_t i = 0; i < number_of_hashes; ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            // a
            hash_functions[d + i * (dim + 1)] = std::abs(rnd_uniform_dist(rnd_normal_gen));
        }
        // b
        hash_functions[dim + i * (dim + 1)] = rnd_uniform_dist(rnd_uniform_gen);
    }
    for (std::size_t i = 0; i < 10; ++i) {
        std::cout << hash_functions[i] << " ";
    }
    std::cout << std::endl;

    std::vector<real_t> data(size * dim);
    std::iota(data.begin(), data.end(), 0);
    std::vector<index_t> hashes(size);
    std::vector<index_t> hash_table(size);

    sycl::device device = sycl::default_selector{}.select_device();
    sycl::queue queue(device);
    std::cout << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << queue.get_device().get_info<sycl::info::device::vendor>() << std::endl;

    const auto previous_power_of_two = [&] (const auto local_mem_size) -> std::size_t {
        return std::pow(2, std::floor(std::log2(local_mem_size / (dim * sizeof(real_t)))));
    };

    auto local_mem_size = queue.get_device().get_info<sycl::info::device::local_mem_size>();
    const auto max_work_item_sizes = queue.get_device().get_info<sycl::info::device::max_work_item_sizes>();
    const auto local_size = std::min(previous_power_of_two(local_mem_size), max_work_item_sizes[0]);
    std::cout << "local size: " << local_size << std::endl;

    const auto round_up = [](const auto x, const auto y) { return ((x + y - 1) / y) * y; };
    const std::size_t global_size = round_up(size, local_size);
    const std::size_t num_group = global_size / local_size;

    auto execution_range = sycl::nd_range<1>{ sycl::range<1>{ global_size }, sycl::range<1>{ local_size } };


    const auto start_time = std::chrono::steady_clock::now();
    {
        sycl::buffer buf_hash_functions(hash_functions.data(), sycl::range<1>{ hash_functions.size() });
        sycl::buffer buf_data(data.data(), sycl::range<1>{ data.size() });
        sycl::buffer buf_hashes(hashes.data(), sycl::range<1>{ hashes.size() });

//        queue.submit([&](sycl::handler& cgh) {
//            auto acc_hf = buf_hash_functions.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);
//            auto acc_d = buf_data.get_access<sycl::access::mode::read>(cgh);
//            auto acc_h = buf_hashes.get_access<sycl::access::mode::discard_write>(cgh);
//
//            sycl::accessor<real_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(sycl::range<1>{ local_size * dim }, cgh);
//            auto execution_range = sycl::nd_range<1>{ sycl::range<1>{ global_size }, sycl::range<1>{ local_size } };
//            cgh.parallel_for<class hash_point>(execution_range, [=](sycl::nd_item<1> item_id) {
//                const std::size_t idx = item_id.get_global_linear_id();
//                const std::size_t local_idx = item_id.get_local_linear_id();
//
//                if (idx >= size) { return; }
//
//                for (std::size_t i = 0; i < dim; ++i) {
////                    local_mem[local_idx + i * local_size] = acc_d[i + idx * dim]; // SoA
//                    local_mem[i + local_idx * dim] = acc_d[i + idx * dim]; // AoS
//                }
//
//                item_id.barrier(sycl::access::fence_space::local_space);
//
//                index_t combined_hash = number_of_hashes;
//                for (std::size_t j = 0; j < number_of_hashes; ++j) {
//                    real_t hash = acc_hf[dim + j * (dim + 1)];
//                    for (std::size_t d = 0; d < dim; ++d) {
////                        hash += local_mem[local_idx + d * local_size] * acc_hf[d + j * (dim + 1)]; // SoA
//                        hash += local_mem[d + local_idx * dim] * acc_hf[d + j * (dim + 1)]; // Aos
//                    }
//                    combined_hash ^= static_cast<index_t>(hash / w) + 0x9e3779b9 + (combined_hash << 6u) + (combined_hash >> 2u);
//                }
//                acc_h[idx] = combined_hash % prim;
//            });
//        });


        {
            options opt;
            hash_tables tables(queue, buf_data, opt);

//            queue.submit([&](sycl::handler& cgh) {
//                auto acc = tables.hash_functions.get_access<sycl::access::mode::read_write>(cgh);
//
//                cgh.parallel_for<class test_kernel>(sycl::range<1>{ tables.hash_functions.get_count() }, [=](sycl::item<1> item) {
//                   const std::size_t idx = item.get_linear_id();
//                   acc[idx] = ++acc[idx];
//                });
//            });
//            queue.wait();

            auto acc = tables.hash_functions_.get_access<sycl::access::mode::read>();
            for (std::size_t i = 0; i < 10; ++i) {
                std::cout << acc[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    queue.wait();

    const auto end_time = std::chrono::steady_clock::now();
    std::cout << "Elapsed Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
              <<  " ms" << std::endl;



    // 71052 29125 23365 29644 45960 98123 89127 79665 34940 51025
//    print(hashes.begin(), hashes.begin() + 10);
//    print(hashes.end() - 10, hashes.end());




    return 0;
}

