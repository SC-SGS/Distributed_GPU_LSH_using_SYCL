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


    std::size_t len = arr.size();
    while (len != 1) {
        // division rounding up
        auto n_wgroups = (len + part_size - 1) / part_size;

        queue.submit([&](sycl::handler& cgh) {
            sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(sycl::range<1>{ wgroup_size }, cgh);

            auto global_mem = buf.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class kernel_reduction>(sycl::nd_range<>(n_wgroups * wgroup_size, wgroup_size), [=](sycl::nd_item<> item) {
                std::size_t local_id = item.get_local_linear_id();
                std::size_t global_id = item.get_global_linear_id();

                local_mem[local_id] = 0;

                if (2 * global_id < len) {
                    local_mem[local_id] = global_mem[2 * global_id] + global_mem[2 * global_id + 1];
                }

                item.barrier(sycl::access::fence_space::local_space);

                for (std::size_t stride = 1; stride < wgroup_size; stride *=2) {
                    std::size_t idx = 2 * stride * local_id;
                    if (idx < wgroup_size) {
                        local_mem[idx] = local_mem[idx] + local_mem[idx + stride];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (local_id == 0) {
                    global_mem[item.get_group_linear_id()] = local_mem[0];
                }

            });

        });
        queue.wait_and_throw();

        len = n_wgroups;
    }


    auto acc = buf.get_access<sycl::access::mode::read>();
    std::cout << "Sum: " << acc[0] << std::endl;

    return 0;
}