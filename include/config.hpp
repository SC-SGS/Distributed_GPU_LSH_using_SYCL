#ifndef DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
#define DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP

#include <cstdint>
#include <CL/sycl.hpp>

using real_t = float;
using integer_t = uint32_t;

namespace sycl = cl::sycl;


enum class memory_type {
    aos, // Array of Structs
    soa  // Struct of Arrays
};

constexpr memory_type layout = memory_type::aos;

#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
