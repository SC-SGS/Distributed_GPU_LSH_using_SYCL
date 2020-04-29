#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP

#include <config.hpp>

template <typename data_type>
class data {
public:
    template <typename Options>
    data(const std::size_t size, const std::size_t dim, const Options& opt)
        : data_(size * dim)
    {
        std::iota(data_.begin(), data_.end(), 0);
    }

    const std::size_t size = 4;
    const std::size_t dims = 3;

    std::vector<data_type> data_;
};

template <typename Options>
data(const std::size_t, const std::size_t, const Options&) -> data<typename Options::real_type>;

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
