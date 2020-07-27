/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-25
 *
 * @brief Implements a conversion function from std::string to numeric types.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERT_HPP

#include <charconv>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <config.hpp>

namespace detail {

    /**
     * @brief Attempt to convert the value represented by @p str to a value of type `T`.
     * @tparam T the value type to parse @p str to (must be arithmetic types)
     * @param[in] str the string to parse
     * @return the value of type `T` represented by @p str (`[[nodiscard]]`)
     *
     * @throw std::invalid_argument if @p str can't get parsed to type `T`.
     */
    template <typename T, REQUIRES(std::is_arithmetic_v<std::remove_reference_t<T>>)>
    [[nodiscard]] inline T convert_to(const std::string& str) {
        using decayed_type = std::decay_t<T>;

        if constexpr (std::is_floating_point_v<decayed_type>) {
            // try to convert str to a floating point type using the old stof/stod/stold functions
            // since std::from_chars doesn't support floating point types yet
            try {
                if constexpr (std::is_same_v<decayed_type, float>) {
                    return std::stof(str);
                } else if constexpr (std::is_same_v<decayed_type, double>) {
                    return std::stod(str);
                } else {
                    return std::stold(str);
                }
            } catch (const std::exception& e) {
                throw std::invalid_argument("Can't parse string '" + str + "'!");
            }
        } else {
            // convert str to integral type using std::from_chars
            T val;
            auto [p, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
            if (ec == std::errc()) {
                return val;
            } else {
                throw std::invalid_argument("Can't parse string '" + str + "'!");
            }
        }
    }

    /**
     * @brief Converts the given value @p val to a [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string).
     * @details Uses the [`operator<<`](https://en.cppreference.com/w/cpp/language/operators) overload.
     * @tparam T the type to convert.
     * @param[in] val the value to convert
     * @return @p val represented as a [`std::string`](https://en.cppreference.com/w/cpp/string/basic_string) (`[[nodiscard]]`)
     */
    template <typename T>
    [[nodiscard]] inline std::string to_string(const T& val) {
        std::ostringstream ss;
        ss << val;
        return ss.str();
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERT_HPP
