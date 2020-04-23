#include <iostream>

#include <CL/sycl.hpp>
#include <numeric>

namespace sycl = cl::sycl;

template <typename Acc>
class constant_adder {
    using T = typename Acc::value_type;
public:

    constant_adder(Acc accessor, const T val, const std::size_t size) : accessor_(accessor), val_(val), size_(size) { }

    void operator()() {
        for (std::size_t i = 0; i < size_; ++i) {
            accessor_[i] += val_;
        }
    }

private:
    Acc accessor_;
    const T val_;
    const std::size_t size_;
};

template <typename Acc>
constant_adder(Acc accessor, typename Acc::value_type, std::size_t) -> constant_adder<decltype(accessor)>;


int main() {
    constexpr std::size_t size = 32;

    std::vector<float> vals(size);
    std::iota(vals.begin(), vals.end(), 0);

    sycl::queue queue(sycl::default_selector{});
    {
        sycl::buffer buf(vals.data(), sycl::range<>{ size });
        queue.submit([&] (sycl::handler& cgh) {
           auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
           cgh.single_task(constant_adder(acc, 2, size));
        });
    }

    for (const auto val : vals) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}