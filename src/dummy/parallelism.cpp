#include <iostream>

#include <CL/sycl.hpp>
#include <numeric>

namespace sycl = cl::sycl;



int main() {

    std::string text("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc interdum in erat non scelerisque.");
    std::cout << text << std::endl;

    sycl::queue queue(sycl::default_selector{});

    {
        sycl::buffer buf(text.data(), sycl::range<>{ text.size() });

        queue.submit([&] (sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class parrot13>(sycl::range<>{ text.size() - 1 }, [=] (sycl::item<1> item) {
                const std::size_t id = item.get_linear_id();
                const auto c = acc[id];
                acc[id] = (c-1/(~(~c|32)/13*2-11)*13);
            });
        });
    }

    std::cout << text << std::endl;
    return 0;
}