/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-06
 *
 * @brief Implements a conversion function from std::string to numeric types.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERT_HPP

#include <charconv>
#include <string>
#include <type_traits>


namespace detail {

    /**
     * @brief Attempt to convert the value represented by @p str to a value of type `T`.
     * @tparam T the value type to parse @p str to (must be arithmetic types)
     * @param[in] str the string to parse
     * @return the value of type `T` represented by @p str
     *
     * @throw std::invalid_argument if @p str can't get parsed to type `T`.
     */
    template <typename T, std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<T>>, int> = 0>
    T convert_to(const std::string& str) {
        if constexpr (std::is_floating_point_v<std::decay_t<T>>) {
            // try to convert str to a floating point type using the old stof/stod/stold functions
            // since std::from_chars doesn't support floating point type syet
            try {
                if constexpr (std::is_same_v<std::decay_t<T>, float>) {
                    return std::stof(str);
                } else if constexpr (std::is_same_v<std::decay_t<T>, double>) {
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

}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERT_HPP
