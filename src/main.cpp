#include <iostream>

#include <CL/sycl.hpp>
#include <numeric>
#include <random>

namespace sycl = cl::sycl;



int main() {
    constexpr std::size_t size = 16;

    std::random_device rnd_device{};
    std::mt19937 rnd_engine(rnd_device());
    std::uniform_int_distribution<int> dist(1, 10);

    std::vector<int> arr(size);
    std::generate(arr.begin(), arr.end(), [&] () { return dist(rnd_engine); });

    std::for_each(arr.begin(), arr.end(), [] (const auto val) { std::cout << val << " "; });
    std::cout << std::endl;

    sycl::buffer buf(arr.data(), sycl::range<>{ size });

    sycl::device device = sycl::default_selector{}.select_device();
    sycl::async_handler exception_handler = [] (sycl::exception_list el) {
        for (auto ex : el) {
            std::rethrow_exception(ex);
        }
    };
    sycl::queue queue(device, exception_handler);

    auto wgroup_size = device.get_info<sycl::info::device::max_work_group_size>();
    std::cout << "wgroup_size: " << wgroup_size << std::endl;
    if (wgroup_size % 2 != 0) {
        throw "Work-group size has to be even!";
    }
    auto part_size = wgroup_size * 2;
    std::cout << "part_size: " << part_size << std::endl;

    auto has_local_mem = device.is_host() || (device.get_info<sycl::info::device::local_mem_type>() != sycl::info::local_mem_type::none);
    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    std::cout << "local_mem_size: " << local_mem_size << " Byte" << std::endl;
    if (!has_local_mem || local_mem_size < (wgroup_size * sizeof(int))) {
        throw "Device doesn't have enough local memory!";
    }






    auto acc = buf.get_access<sycl::access::mode::read>();
    std::cout << "Sum: " << acc[0] << std::endl;

    return 0;
}