/**
 * @brief
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_PARSER_HPP


#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <config.hpp>
#include <detail/convert.hpp>
#include <file_parser/file_parser.hpp>


template <memory_layout layout, typename Options>
class default_parser final : public file_parser<layout, Options> {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");

    using base = file_parser<layout, Options>;
public:
    using real_type = typename Options::real_type;
    using index_type = typename Options::index_type;

    // TODO 2020-06-02 18:18 marcel: private + friend
    explicit default_parser(std::string file) : file_parser<layout, Options>(std::move(file)) {
        std::cout << "Parsing a file in .txt format using the default_parser!" << std::endl;
    }

    [[nodiscard]] index_type parse_size() const override {
        std::ifstream in(base::file_);
        return std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n');
    }
    [[nodiscard]] index_type parse_dims() const override {
        std::ifstream in(base::file_);
        std::string line;
        std::getline(in, line);
        return line.empty() ? 0 : std::count(line.cbegin(), line.cend(), ',') + 1;
    }
    void parse_content(sycl::buffer<real_type, 1>& buffer, const index_type size, const index_type dims) const override {
        std::ifstream in(base::file_);
        std::string line, elem;

        // read file line by line, parse value and save it at the correct position (depending on the current memory_layout) in buffer
        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        for (index_type point = 0; point < size; ++point) {
            std::getline(in, line);
            std::stringstream ss(line);
            for (index_type dim = 0; dim < dims; ++dim) {
                std::getline(ss, elem, ',');
                acc[base::get_linear_id(point, dim, size, dims)] = detail::convert_to<real_type>(elem);
            }
        }
    }
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_PARSER_HPP
