/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Different utility functions or macros.
 */

#ifndef SYCL_LSH_DETAIL_UTILITY_HPP
#define SYCL_LSH_DETAIL_UTILITY_HPP
#pragma once

#include "sycl_lsh/exceptions/exceptions.hpp"  // sycl_lsh::exception

#include "sycl/sycl.hpp"

#include "fmt/format.h"  // fmt::format

#include <charconv>      // std::form_char
#include <string>        // std::string, std::stof, std::stod, std::stold
#include <string_view>   // std::string_view
#include <system_error>  // std::errc
#include <type_traits>   // std::enable_if_t, std::is_arithmetic_v, std::is_same_v

/**
 * @brief Helper function for an extra round of macro expansion inside the  SYCL_LSH_IS_DEFINED macro.
 */
#define SYCL_LSH_IS_DEFINED_HELPER(x) #x
/**
 * @brief Evaluates to `true` if the preprocessor macro @p x is defined, otherwise `false`.
 * @details Based on: https://stackoverflow.com/questions/18048039/c-constexpr-function-to-test-preprocessor-macros
 */
#define SYCL_LSH_IS_DEFINED(x) (std::string_view{ #x } != std::string_view{ SYCL_LSH_IS_DEFINED_HELPER(x) })

/**
 * @brief Shorthand macro for an easier [std::enable_if_t](https://en.cppreference.com/w/cpp/types/enable_if).
 */
#define SYCL_LSH_REQUIRES(...) std::enable_if_t<(__VA_ARGS__), bool> = true

namespace sycl_lsh::detail {

/**
 * @brief Type-dependent expression that always evaluates to `false`.
 */
template <typename>
constexpr bool always_false_v = false;

/**
 * @brief Value-dependent expression that always evaluates to `false`.
 */
template <auto>
constexpr bool always_false_non_type_v = false;

/**
 * @brief Invokes undefined behavior. Used to mark code paths that may never be reachable.
 * @details See: C++23 [std::unreachable](https://en.cppreference.com/w/cpp/utility/unreachable)
 */
[[noreturn]] inline void unreachable() {
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
    #if defined(__GNUC__)  // GCC, Clang, ICC
    __builtin_unreachable();
    #elif defined(_MSC_VER)  // MSVC
    __assume(false);
    #endif
}

/**
 * A shorthand alias for a general sycl::atomic_ref.
 */
template <typename T>
using atomic_op = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;

/**
 * @brief Return a new string with the same content as @p str but all lower case.
 * @param[in] str the string to use in the transformation
 * @return the transformed string (`[[nodiscard]]`)
 */
[[nodiscard]] std::string to_lower_case(std::string_view str);

/**
 * @brief Checks whether the string @p str contains the string @p substr.
 * @param[in] str the string that contains the sub-string
 * @param[in] substr the sub-string to find
 * @return `true` if @p str contains @p substr, `false` otherwise (`[[nodiscard]]`)
 */
[[nodiscard]] bool contains_substr(std::string_view str, std::string_view substr) noexcept;

/**
 * @brief Attempt to convert the value represented by the
 *        [std::string](https://en.cppreference.com/w/cpp/string/basic_string) @p str to a value if type `T`.
 * @tparam T the type to which convert the given string
 * @param[in] str the string to convert
 * @return the value if type `T` represented by @p str (`[[nodiscard]]`)
 *
 * @throw sycl_lsh::exception if @p str can't be converted to type `T`.
 */
template <typename T, SYCL_LSH_REQUIRES(std::is_arithmetic_v<T>)>
[[nodiscard]] T convert_to(const std::string &str) {
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
        }
    }
    throw exception{ fmt::format("Can't convert '{}' to the requested type!", str) };
}

}  // namespace sycl_lsh::detail

#endif  // SYCL_LSH_DETAIL_UTILITY_HPP
