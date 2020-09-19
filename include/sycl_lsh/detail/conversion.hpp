/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-19
 *
 * @brief Implements a conversion function from std::string to arithmetic types.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERSION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERSION_HPP

#include <sycl_lsh/detail/utility.hpp>

#include <charconv>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace sycl_lsh::detail {

    /**
     * @brief Attempt to convert the value represented by the
     *        [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string) @p str to a value if type `T`.
     * @tparam T the type to which convert the given string
     * @param[in] str the string to convert
     * @return the value if type `T` represented by @p str (`[[nodiscard]]`)
     *
     * @throw std::invalid_argument if @p str can't be converted to type `T`.
     */
    template <typename T, SYCL_LSH_REQUIRES(std::is_arithmetic_v<T>)>
    [[nodiscard]]
    inline T convert_to(const std::string& str) {
        if constexpr (std::is_floating_point_v<T>) {
            // convert floating point numbers using stof, stod or stold respectively
            if constexpr (std::is_same_v<T, float>) {
                return std::stof(str);
            } else if constexpr (std::is_same_v<T, double>) {
                return std::stod(str);
            } else {
                return std::stold(str);
            }
        } else {
            // convert integral numbers using C++17 std::from_chars
            T val;
            auto [p, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
            if (ec == std::errc()) {
                // no error occurred during conversion
                return val;
            } else {
                throw std::invalid_argument("Can't convert '" + str + "' to the requested type!");
            }
        }
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERSION_HPP
