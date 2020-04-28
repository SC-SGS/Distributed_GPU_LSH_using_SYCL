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

//    std::random_device gaussian_generator;
//    std::mt19937 mt1(gaussian_generator());
    std::mt19937 mt1;
    std::normal_distribution<real_t> gaussian_distribution;
//    std::random_device uniform_generator;
//    std::mt19937 mt2(uniform_generator());
    std::mt19937 mt2;
    std::uniform_real_distribution<real_t> uniform_distribution(0, w);

    for (std::size_t i = 0; i < number_of_hashes; ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            // a
            hash_functions[d + i * (dim + 1)] = std::abs(gaussian_distribution(mt1));
        }
        // b
        hash_functions[dim + i * (dim + 1)] = uniform_distribution(mt2);
    }

    print(hash_functions.begin(), hash_functions.end());

    std::vector<real_t> data(size * dim);
    std::iota(data.begin(), data.end(), 0);
    std::vector<index_t> hashes(size);

    // for each data point
    for (std::size_t i = 0; i < size; ++i) {
        // for each hash function
        index_t combined_hash = number_of_hashes;
        for (std::size_t j = 0; j < number_of_hashes; ++j) {
            // calculate (a * v + b) / w
            real_t hash = hash_functions[dim + j * (dim + 1)];
            for (std::size_t d = 0; d < dim; ++d) {
                hash += data[d + i * dim] * hash_functions[d + j * (dim + 1)];
            }
            // combine hashes
            combined_hash ^= static_cast<index_t>(hash / w) + 0x9e3779b9 + (combined_hash << 6u) + (combined_hash >> 2u);
        }
        // write back hashes
        hashes[i] = combined_hash % prim;
    }

    print(hashes.begin(), hashes.end());

    return 0;
}