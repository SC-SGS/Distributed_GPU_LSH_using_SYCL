/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-03
 *
 * @brief File parser for parsing plain data files.
 */


#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_PARSER_HPP


#include <iostream>
#include <fstream>
#include <string>

#include <config.hpp>
#include <detail/convert.hpp>
#include <file_parser/base_file_parser.hpp>


/**
 * @brief Default file parser for parsing plain data files.
 * @tparam layout determines whether the data is saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 */
template <memory_layout layout, typename Options>
class default_parser final : public file_parser<layout, Options> {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");

    /// The type of the base @ref file_parser.
    using base = file_parser<layout, Options>;
public:
    /// The type of the underlying data as specified as in the provided @ref options class.
    using real_type = typename Options::real_type;
    /// The index type as specified as in the provided @ref options class.
    using index_type = typename Options::index_type;

    /**
     * @brief Constructs a new @ref default_parser object for parsing plain data files.
     * @param[in] file the file to parse
     *
     * @throw std::invalid_argument if @p file doesn't exist
     */
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
