#ifndef DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
#define DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP


#include <CL/sycl.hpp>


namespace sycl = cl::sycl;


enum class memory_layout {
    aos, // Array of Structs
    soa  // Struct of Arrays
};


#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
