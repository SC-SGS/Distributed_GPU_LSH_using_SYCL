/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-28
 *
 * @brief Header to add a namespace alias for the sycl implementations.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SYCL_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SYCL_HPP

#include <CL/sycl.hpp>

namespace sycl_lsh {

    namespace sycl = cl::sycl;

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SYCL_HPP
