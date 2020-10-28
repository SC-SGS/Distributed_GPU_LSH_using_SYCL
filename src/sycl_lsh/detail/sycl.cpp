/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 */

#include <sycl_lsh/detail/sycl.hpp>

#include <exception>
#include <iostream>

void sycl_lsh::sycl_exception_handler(const sycl_lsh::sycl::exception_list exceptions) {
    for (const std::exception_ptr& e_ptr : exceptions) {
        try {
            std::rethrow_exception(e_ptr);
        } catch (const sycl::exception& e) {
            std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
        }
    }
}