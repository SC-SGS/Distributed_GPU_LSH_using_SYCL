/**
 * @file src/main.cpp
 * @author Marcel Breyer
 * @date 2020-04-18
 */

#include <cassert>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>
#include <mpi.h>

using data_type = float;

std::vector<data_type> add(cl::sycl::queue& q, const std::vector<data_type>& a, const std::vector<data_type>& b) {
    std::vector<data_type> c(a.size());

    assert(a.size() == b.size());
    cl::sycl::range<1> work_items{a.size()};

    {
        cl::sycl::buffer<data_type> buff_a(a.data(), a.size());
        cl::sycl::buffer<data_type> buff_b(b.data(), b.size());
        cl::sycl::buffer<data_type> buff_c(c.data(), c.size());

        q.submit([&](cl::sycl::handler& cgh){
            auto access_a = buff_a.get_access<cl::sycl::access::mode::read>(cgh);
            auto access_b = buff_b.get_access<cl::sycl::access::mode::read>(cgh);
            auto access_c = buff_c.get_access<cl::sycl::access::mode::write>(cgh);

            cgh.parallel_for<class vector_add>(work_items,
                                               [=] (cl::sycl::id<1> tid) {
                                                   access_c[tid] = access_a[tid] + access_b[tid];
                                               });
        });
    }
    return c;
}

int main() {
    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Rank: " << rank << std::endl;

    cl::sycl::queue q;
    std::vector<data_type> a = { 1.f, 2.f, 3.f, 4.f, 5.f };
    std::vector<data_type> b = { -1.f, 2.f, -3.f, 4.f, -5.f };
    auto result = add(q, a, b);

    std::cout << "Result: " << std::endl;
    for (const auto x: result) {
        std::cout << x << std::endl;
    }

    MPI_Finalize();
    return 0;
}