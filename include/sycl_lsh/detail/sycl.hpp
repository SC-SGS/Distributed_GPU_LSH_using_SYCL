/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-07
 *
 * @brief Header to add a namespace alias for the sycl implementations and an exception handler.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SYCL_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SYCL_HPP

#include <CL/sycl.hpp>

#include <exception>
#include <iostream>

namespace sycl_lsh {

    namespace sycl = cl::sycl;

    /**
     * @brief Exception handler called on `queue.wait_and_throw()` if an exception was thrown inside a SYCL kernel.
     * @param[in] exceptions list of thrown exceptions
     */
    void sycl_exception_handler(sycl::exception_list exceptions) {
        for (const std::exception_ptr& e_ptr : exceptions) {
            try {
                std::rethrow_exception(e_ptr);
            } catch (const sycl::exception& e) {
                std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
            }
        }
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SYCL_HPP
