#include <iostream>

#include <CL/sycl.hpp>
#include <numeric>

namespace sycl = cl::sycl;

int main() {
    constexpr std::size_t size = 10;

    std::vector<float> a(size);
    std::iota(a.begin(), a.end(), 0);
    std::vector<float> b(size);
    std::iota(b.begin(), b.end(), 0);
    std::vector<float> c(size, 0);

    sycl::default_selector device_selector;

    sycl::queue queue(device_selector);
    std::cout << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << queue.get_device().get_info<sycl::info::device::vendor>() << std::endl;

    {
        // NEVER EVER USE begin(), end() constructor!!!
        sycl::buffer<float> a_sycl(a.data(), sycl::range<>{ size });
        sycl::buffer<float> b_sycl(b.data(), sycl::range<>{ size });
        sycl::buffer<float> c_sycl(c.data(), sycl::range<>{ size });

        queue.submit([&] (sycl::handler& cgh) {
           auto a_acc = a_sycl.get_access<sycl::access::mode::read>(cgh);
           auto b_acc = b_sycl.get_access<sycl::access::mode::read>(cgh);
           auto c_acc = c_sycl.get_access<sycl::access::mode::discard_write>(cgh);

           cgh.parallel_for<class vector_addition>(sycl::range<>{ size }, [=] (sycl::item<> idx) {
               c_acc[idx] = a_acc[idx] + b_acc[idx];
           });
        });
    }

    for (std::size_t i = 0; i < size; ++i) {
        std::cout << c[i] << " = " << a[i] << " + " << b[i] << std::endl;
    }

    return 0;
}