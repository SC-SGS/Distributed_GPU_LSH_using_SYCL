#include <iostream>

#include <CL/sycl.hpp>
#include <random>
#include <cmath>
#include <cassert>
#include <type_traits>

namespace sycl = cl::sycl;


template <typename InputIt>
void print(InputIt first, InputIt last) {
    std::for_each(first, last, [](const auto val) { std::cout << val << " "; });
    std::cout << std::endl;
}

using real_t = float;
using index_t = uint32_t;

int main() {
    constexpr std::size_t dim = 16;
    constexpr std::size_t number_of_hashes = 16;
    // w
    constexpr real_t w = 1.0;
    constexpr index_t prim = 105613;
    constexpr std::size_t size = 1000;

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

    std::vector<real_t> data(size * dim);
    std::iota(data.begin(), data.end(), 0);
    std::vector<index_t> hashes(size);

    sycl::device device = sycl::default_selector{}.select_device();
    sycl::queue queue(device);
    std::cout << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << queue.get_device().get_info<sycl::info::device::vendor>() << std::endl;

    auto wgroup_size = device.get_info<sycl::info::device::max_work_group_size>();

    const auto start_time = std::chrono::steady_clock::now();
    {
        sycl::buffer buf_hash_functions(hash_functions.data(), sycl::range<1>{ hash_functions.size() });
        sycl::buffer buf_data(data.data(), sycl::range<1>{ data.size() });
        sycl::buffer buf_hashes(hashes.data(), sycl::range<1>{ hashes.size() });

        queue.submit([&](sycl::handler& cgh) {
            auto acc_hf = buf_hash_functions.get_access<sycl::access::mode::read>(cgh);
            auto acc_d = buf_data.get_access<sycl::access::mode::read>(cgh);
            auto acc_h = buf_hashes.get_access<sycl::access::mode::discard_write>(cgh);

            const std::size_t local_size = 128;
            const auto round_up = [](const auto x, const auto y) { return ((x + y - 1) / y) * y; };
            const std::size_t global_size = round_up(size, local_size);
            const std::size_t num_group = global_size / local_size;

            std::cout << "local_size: " << local_size << std::endl;
            std::cout << "global_size: " << global_size << std::endl;
            std::cout << "num_group: " << num_group << std::endl;

            auto execution_range = sycl::nd_range<1>{ sycl::range<1>{ global_size }, sycl::range<1>{ local_size } };
            std::cout << execution_range.get_global_range()[0] << " " << execution_range.get_local_range()[0] << std::endl;

            cgh.parallel_for<class hash_point>(execution_range, [=](sycl::nd_item<1> item_id) {
                const std::size_t idx = item_id.get_global_linear_id();

                const std::size_t global_id = item_id.get_global_id(0);
                const std::size_t local_id = item_id.get_local_id(0);

                if (global_id == 0) {
                    printf(idx == global_id ? "true\n" : "false\n");
                }


                if (global_id == 0 || global_id == 64 || global_id == 127 || global_id == 128 || global_id == 129 || global_id == 256) {
                    printf("%lu -> %lu (%lu) -> %lu \n", global_id, local_id, item_id.get_group(0), idx);

                }

                if (idx >= size) {
                    if (idx == size) printf("Oversubscribed\n");
                    return;
                }

                index_t combined_hash = number_of_hashes;
                for (std::size_t j = 0; j < number_of_hashes; ++j) {
                    real_t hash = acc_hf[dim + j * (dim + 1)];
                    for (std::size_t d = 0; d < dim; ++d) {
                        hash += acc_d[d + idx * dim] * acc_hf[d + j * (dim + 1)];
                    }
                    combined_hash ^= static_cast<index_t>(hash / w) + 0x9e3779b9 + (combined_hash << 6u) + (combined_hash >> 2u);
                }
                acc_h[idx] = combined_hash % prim;
            });
        });
    }
    queue.wait();

    const auto end_time = std::chrono::steady_clock::now();
    std::cout << "Elapsed Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
              <<  " ms" << std::endl;

    // 71052 29125 23365 29644 45960 98123 89127 79665 34940 51025
    print(hashes.begin(), hashes.begin() + 10);
    print(hashes.end() - 10, hashes.end());


    return 0;
}

