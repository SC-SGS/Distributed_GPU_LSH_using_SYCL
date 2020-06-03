/**
 * @brief
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP


#include <string>
#include <iostream>
#include <vector>

#include <config.hpp>
#include <detail/convert.hpp>
#include <file_parser/base_file_parser.hpp>


template <memory_layout layout, typename Options>
class arff_parser final : public file_parser<layout, Options> {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");

    using base = file_parser<layout, Options>;
public:
    using real_type = typename Options::real_type;
    using index_type = typename Options::index_type;

    explicit arff_parser(std::string file) : file_parser<layout, Options>(std::move(file)) {
        std::cout << "Parsing a file in .arff format using the arff_parser!" << std::endl;
    }

    [[nodiscard]] index_type parse_size() const override {
        std::ifstream in(base::file_);
        // skip until @DATA tag read
        std::string line;
        while (std::getline(in, line)) {
            if (this->starts_with(line, "@DATA")) break;
        }
        // count lines from here on
        return std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n');
    }
    [[nodiscard]] index_type parse_dims() const override {
        std::ifstream in(base::file_);
        index_type dims = 0;
        // count number of @ATTRIBUTES until @DATA
        std::string line;
        while (std::getline(in , line)) {
            if (starts_with(line, "@ATTRIBUTE")) {
                ++dims;
            } else if (starts_with(line, "@DATA")) {
                break;
            }
        }
        return dims;
    }
    void parse_content(sycl::buffer<real_type, 1>& buffer, const index_type size, const index_type dims) const override {
        std::ifstream in(base::file_);
        // skip until @DATA tag read
        std::string line, elem;
        while (std::getline(in, line)) {
            if (this->starts_with(line, "@DATA")) break;
        }

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

private:
    [[nodiscard]] bool starts_with(const std::string_view str, const std::string_view start) const noexcept {
        return str.compare(0, start.size(), start) == 0;
    }
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP
