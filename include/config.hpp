#ifndef DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
#define DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP


#include <CL/sycl.hpp>


namespace sycl = cl::sycl;


enum class memory_type {
    aos, // Array of Structs
    soa  // Struct of Arrays
};

constexpr memory_type layout = memory_type::aos;

#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
