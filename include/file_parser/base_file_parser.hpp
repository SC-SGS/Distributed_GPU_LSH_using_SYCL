/**
 * @brief
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_FILE_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_FILE_PARSER_HPP


#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <config.hpp>
#include <detail/assert.hpp>
#include <options.hpp>


template <memory_layout layout, typename Options>
class base_file_parser {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");
public:
    using real_type = typename Options::real_type;
    using index_type = typename Options::index_type;

    explicit base_file_parser(std::string file) : file_(std::move(file)) { }
    virtual ~base_file_parser() = default;

    /**
     * @brief Computes the number of data points in the file.
     * @return the number of data points (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual index_type parse_size() const = 0;
    /**
     * @brief Computes the number of dimensions of each data point in the file.
     * @return the number of dimensions (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual index_type parse_dims() const = 0;
    virtual void parse_content(sycl::buffer<real_type, 1>& buffer, const index_type size, const index_type dims) const = 0;

protected:
    [[nodiscard]] constexpr index_type get_linear_id(const index_type point, const index_type dim,
            const index_type size, const index_type dims) const noexcept
    {
        DEBUG_ASSERT(0 <= point && point < size, "Out-of-bounce access!: 0 <= {} < {}", point, size);
        DEBUG_ASSERT(0 <= dim && dim < dims, "Out-of-bounce access!: 0 <= {} < {}", dim, dims);

        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return dim + point * dims;
        } else {
            // Struct of Arrays
            return point + dim * size;
        }
    }

    const std::string file_;
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_FILE_PARSER_HPP
