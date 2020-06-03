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
class arff_parser final : public base_file_parser<layout, Options> {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");

    using base = base_file_parser<layout, Options>;
public:
    using real_type = typename Options::real_type;
    using index_type = typename Options::index_type;

    explicit arff_parser(std::string file) : base_file_parser<layout, Options>(std::move(file)) {
        std::cout << "Parsing a file in .arff format using the arff_parser!" << std::endl;
    }

    [[nodiscard]] index_type parse_size() const override { throw std::logic_error("Not implemented yet!"); }
    [[nodiscard]] index_type parse_dims() const override { throw std::logic_error("Not implemented yet!"); }
    void parse_content(sycl::buffer<real_type, 1>& buffer, const index_type size, const index_type dims) const override {
        throw std::logic_error("Not implemented yet!");
    }
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP
