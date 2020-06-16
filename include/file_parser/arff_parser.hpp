/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-16
 *
 * @brief File parser for parsing `.arff` files.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP


#include <iostream>
#include <string>
#include <type_traits>

#include <config.hpp>
#include <detail/convert.hpp>
#include <file_parser/base_file_parser.hpp>


/**
 * @brief File parser for parsing `.arff` files.
 * @tparam layout determines whether the data is saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 */
template <memory_layout layout, typename Options>
class arff_parser final : public file_parser<layout, Options> {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by an 'options' type!");

    /// The type of the base @ref file_parser.
    using base = file_parser<layout, Options>;
public:
    /// The type of the underlying data as specified as in the provided @ref options class.
    using real_type = typename Options::real_type;
    /// The index type as specified as in the provided @ref options class.
    using index_type = typename Options::index_type;

    /**
     * @brief Constructs a new @ref arff_parser object for parsing `.arff` files.
     * @param[in] file the file to parse
     *
     * @throw std::invalid_argument if @p file doesn't exist
     */
    explicit arff_parser(std::string file) : file_parser<layout, Options>(std::move(file), false) {
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
        DEBUG_ASSERT(0 < size, "Illegal size!: {}", size);
        DEBUG_ASSERT(0 < dims, "Illegal number of dimensions!: {}", dims);

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
    /**
     * @brief Checks whether @p str starts with the string @p start.
     * @param str the string which should start with @p start
     * @param start the string to check
     * @return `true` if @p str starts with the string @p start, `false` otherwise
     */
    [[nodiscard]] bool starts_with(const std::string_view str, const std::string_view start) const noexcept {
        return str.compare(0, start.size(), start) == 0;
    }
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP
