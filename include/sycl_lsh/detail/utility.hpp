/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 *
 * @brief Different utility functions or macros.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

/**
 * @brief Shorthand macro for an easier [`std::enable_if`](https://en.cppreference.com/w/cpp/types/enable_if).
 */
#define SYCL_LSH_REQUIRES(cond) std::enable_if_t<(cond), int> = 0

namespace sycl_lsh::detail {

    /**
     * @brief Swaps the values of @p lhs and @p rhs.
     * @tparam T the type of the elements to swap
     * @param[in,out] lhs the first value to swap
     * @param[in,out] rhs the second value to swap
     */
    template <typename T>
    inline void swap(T& lhs, T& rhs) {
        const T tmp = lhs;
        lhs = rhs;
        rhs = tmp;
    }

    /**
     * @brief Checks whether the string @p str contains the string @p substr.
     * @param[in] str the string that contains the sub-string
     * @param[in] substr the sub-string to find
     * @return `true` if @p str contains @p substr, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] 
    inline bool contains_substr(const std::string_view str, const std::string_view substr) noexcept {
        return str.find(substr) != std::string_view::npos;
    }

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

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP
