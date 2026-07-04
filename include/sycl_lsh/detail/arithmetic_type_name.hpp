/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements conversion functions from arithmetic types to their name as string representation.
 */

#ifndef SYCL_LSH_DETAIL_ARITHMETIC_TYPE_NAME_HPP
#define SYCL_LSH_DETAIL_ARITHMETIC_TYPE_NAME_HPP
#pragma once

#include <string_view>  // std::string_view

/**
 * @def SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME
 * @brief Defines a macro to create all possible conversion functions from arithmetic types to their name as string representation.
 * @details Also supports `const` and/or `volatile` qualifiers.
 * @param[in] type the data type to convert to a string
 */
#define SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(type)                                                               \
    template <>                                                                                                  \
    [[nodiscard]] constexpr std::string_view arithmetic_type_name<type>() { return #type; }                      \
    template <>                                                                                                  \
    [[nodiscard]] constexpr std::string_view arithmetic_type_name<const type>() { return "const " #type; }       \
    template <>                                                                                                  \
    [[nodiscard]] constexpr std::string_view arithmetic_type_name<volatile type>() { return "volatile " #type; } \
    template <>                                                                                                  \
    [[nodiscard]] constexpr std::string_view arithmetic_type_name<const volatile type>() { return "const volatile " #type; }

namespace sycl_lsh::detail {

/**
 * @brief Tries to convert the given type to its name as string representation including possible `const` and/or `volatile` qualifiers.
 * @details The definition is marked as **deleted** if `T` isn't an [arithmetic type](https://en.cppreference.com/w/cpp/types/is_arithmetic).
 * @tparam T the type to convert to a string
 * @return the name of `T` (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr std::string_view arithmetic_type_name() = delete;

SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(bool)

// character types
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(char)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(signed char)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(unsigned char)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(char16_t)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(char32_t)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(wchar_t)

// integer types
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(short)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(unsigned short)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(int)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(unsigned int)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(long)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(unsigned long)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(long long)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(unsigned long long)

// floating point types
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(float)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(double)
SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(long double)

}  // namespace sycl_lsh::detail

#undef SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME

#endif  // SYCL_LSH_DETAIL_ARITHMETIC_TYPE_NAME_HPP
