#include <iostream>

#include <CL/sycl.hpp>
#include <numeric>

namespace sycl = cl::sycl;

int main() {

    auto exception_handler = [] (sycl::exception_list exceptions) {
        std::cout << "ERROR" << std::endl;
        for (const std::exception_ptr& ptr : exceptions) {
            try {
                std::rethrow_exception(ptr);
            } catch (const sycl::exception& e) {
                std::cerr << "Asynchronous SYCL exception: " << e.what() << std::endl;
            }
        }
    };

    sycl::queue queue(sycl::default_selector{}, exception_handler, sycl::property_list{});


    queue.submit([&] (sycl::handler& cgh) {
        auto range = sycl::nd_range<1>(sycl::range<1>{ 1 }, sycl::range<1>{ 10 });

        cgh.parallel_for<class vector_addition>(range, [=] (sycl::nd_item<> idx) { });
    });


    try {

        queue.wait_and_throw();
    } catch (const sycl::exception& e) {
        std::cerr << "Synchronous SYCL exception: " << e.what() << std::endl;
    }

    return 0;
}