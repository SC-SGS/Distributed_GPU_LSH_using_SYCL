#include <iostream>

#include <CL/sycl.hpp>
#include <random>
#include <cmath>

namespace sycl = cl::sycl;

template <typename InputIt>
void print(InputIt first, InputIt last) {
    std::for_each(first, last, [](const auto val) { std::cout << val << " "; });
    std::cout << std::endl;
}

using real_t = float;
using index_t = uint32_t;

int main() {
    constexpr std::size_t dim = 6;
    constexpr std::size_t number_of_hashes = 4;
    // w
    constexpr real_t w = 1.0;
    constexpr index_t prim = 105613;
    constexpr std::size_t size = 10;

    std::vector<real_t> hash_functions(number_of_hashes * (dim + 1));
    std::vector<real_t> data(size * dim);
    std::iota(data.begin(), data.end(), 0);
    std::vector<index_t> hashes(size);

    sycl::queue queue(sycl::default_selector{});
    {
        sycl::buffer buf_hash_functions(hash_functions.data(), sycl::range<1>{ hash_functions.size() });

        queue.submit([&](sycl::handler& cgh) {
            auto acc = buf_hash_functions.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class create_hash_functions>(sycl::range<1>{ number_of_hashes }, [=](sycl::item<1> item) {
                const std::size_t idx = item.get_linear_id();

                std::random_device rnd_device;
                std::mt19937 rnd_normal_gen(rnd_device());
                std::normal_distribution<real_t> rnd_normal_dist;
                std::mt19937 rnd_uniform_gen(rnd_device());
                std::uniform_real_distribution<real_t> rnd_uniform_dist(0, w);

                for (std::size_t i = 0; i < dim; ++i) {
                    acc[i + idx * (dim + 1)] = std::abs(rnd_normal_dist(rnd_normal_gen)); // todo: abs?
                }
                acc[dim + idx * (dim + 1)] = rnd_uniform_dist(rnd_uniform_gen);
            });
        });

        sycl::buffer buf_data(data.data(), sycl::range<1>{ data.size() });
        sycl::buffer buf_hashes(hashes.data(), sycl::range<1>{ hashes.size() });

        queue.submit([&](sycl::handler& cgh) {
            auto acc_hf = buf_hash_functions.get_access<sycl::access::mode::read>(cgh);
            auto acc_d = buf_data.get_access<sycl::access::mode::read>(cgh);
            auto acc_h = buf_hashes.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<class hash_point>(sycl::range<1>{ size }, [=](sycl::item<1> item) {
                const std::size_t idx = item.get_linear_id();
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

    print(hashes.begin(), hashes.end());

    return 0;
}