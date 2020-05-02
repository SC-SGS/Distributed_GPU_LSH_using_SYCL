#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP

#include <config.hpp>
#include <detail/convert.hpp>
#include <options.hpp>

#include <algorithm>
#include <fstream>
#include <iterator>


template <memory_layout layout, typename Options>
class data {
public:
    using data_type = typename Options::real_type;
    using size_type = typename Options::size_type;
    using options_type = Options;


    const size_type size;
    const size_type dims;
    sycl::buffer<data_type, 1> buffer;


    template <memory_layout other_layout>
    data(data<other_layout, Options>& other)
        : size(other.size), dims(other.dims), buffer(sycl::range<1>{ other.buffer.get_count() })
    {
        auto acc_this = buffer.template get_access<sycl::access::mode::discard_write>();
        auto acc_other = other.buffer.template get_access<sycl::access::mode::read>();
        for (size_type s = 0; s < size; ++s) {
            for (size_type d = 0; d < dims; ++d) {
                acc_this[this->get_linear_id(s, d)] = acc_other[other.get_linear_id(s, d)];
            }
        }
    }


    [[nodiscard]] size_type get_linear_id(const size_type point, const size_type dim) const noexcept {
        if constexpr (layout == memory_layout::aos) {
            return dim + point * dims;
        } else {
            return point + dim * size;
        }
    }

    [[nodiscard]] constexpr memory_layout get_memory_layout() const noexcept {
        return layout;
    }

private:
    template <memory_layout layout_, typename Options_, typename... Args_>
    friend data<layout_, Options_> make_data(const Options_& opt, Args_&&... args);

    data(const std::size_t size, const std::size_t dims, const Options&)
            : size(size), dims(dims), buffer(sycl::range<1>{ size * dims })
    {
        std::random_device rnd_device;
        std::mt19937 rnd_gen(rnd_device());
        std::normal_distribution<data_type> rnd_dist;

        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        for (size_type i = 0; i < size * dims; ++i) {
            acc[i] = rnd_dist(rnd_gen);
        }
    }
    data(const std::string& file, const Options& opt)
            : size(this->parse_size(file)), dims(this->parse_dims(file)), buffer(sycl::range<1>{ size * dims })
    {
        std::ifstream in(file);
        std::string line, elem;

        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        for (size_type point = 0; point < size; ++point) {
            std::getline(in, line);
            std::stringstream ss(line);
            for (size_type dim = 0; dim < dims; ++dim) {
                std::getline(ss, elem, ',');
                acc[point + dim * size] = detail::convert_to<data_type>(elem);
            }
        }
    }


    [[nodiscard]] size_type parse_size(const std::string& file) const {
        std::ifstream in(file);
        return std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n');
    }
    [[nodiscard]] size_type parse_dims(const std::string& file) const {
        if (size == 0) return 0;

        std::ifstream in(file);
        std::string line;
        std::getline(in, line);
        return std::count(line.cbegin(), line.cend(), ',') + 1;
    }
};


template <memory_layout layout, typename Options, typename... Args>
data<layout, Options> make_data(const Options& opt, Args&&... args) {
    return data<layout, Options>(std::forward<Args>(args)..., opt);
}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
