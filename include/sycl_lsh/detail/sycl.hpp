/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 *
 * @brief Header to add a namespace alias for the sycl implementations and an exception handler.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SYCL_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SYCL_HPP

#include <CL/sycl.hpp>

namespace sycl_lsh {

    namespace sycl = cl::sycl;

    /**
     * @brief Exception handler called on `queue.wait_and_throw()` if an exception was thrown inside a SYCL kernel.
     * @param[in] exceptions list of thrown exceptions
     */
    void sycl_exception_handler(sycl::exception_list exceptions);

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SYCL_HPP
