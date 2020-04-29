#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP

#include <config.hpp>
#include <options.hpp>
#include <detail/convert.hpp>
#include <iterator>
#include <fstream>

template <typename data_type, typename size_type>
class data {
public:
    template <typename Options>
    data(const std::size_t size, const std::size_t dims, const Options&)
        : size(size), dims(dims), data_(sycl::range<1>{ size * dims })
    {
        std::random_device rnd_device;
        std::mt19937 rnd_gen(rnd_device());
        std::normal_distribution<data_type> rnd_dist;

        auto acc = data_.template get_access<sycl::access::mode::discard_write>();
        for (size_type i = 0; i < data_.get_count(); ++i) {
            acc[i] = rnd_dist(rnd_gen);
        }
    }

    template <typename Options>
    data(const std::string& file, const Options& opt)
        : size(this->parse_size(file)), dims(this->parse_dims(file)), data_(sycl::range<1>{ size * dims })
    {
        std::ifstream in(file);
        std::string line, elem;

        auto acc = data_.template get_access<sycl::access::mode::discard_write>();
        for (size_type point = 0; point < size; ++point) {
            std::getline(in, line);
            std::stringstream ss(line);
            for (size_type dim = 0; dim < dims; ++dim) {
                std::getline(ss, elem, ',');
                acc[point + dim * size] = detail::convert_to<data_type>(elem);
            }
        }
    }



    const size_type size = 4;
    const size_type dims = 3;

//private:
    sycl::buffer<data_type, 1> data_;

private:
    size_type parse_size(const std::string& file) const {
        std::ifstream in(file);
        return std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n');
    }
    size_type parse_dims(const std::string& file) const {
        if (size == 0) return 0;

        std::ifstream in(file);
        std::string line;
        std::getline(in, line);
        size_type count;
        for (std::size_t i = 0; i < line.size(); ++i) {
            if (line[i] == ',') {
                ++count;
            }
        }
        return count + 1;
    }
};

template <typename Options>
data(const std::size_t, const std::size_t, const Options&) -> data<typename Options::real_type, typename Options::size_type>;
template <typename Options>
data(const std::string_view, const Options&) -> data<typename Options::real_type, typename Options::size_type>;

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_HPP
