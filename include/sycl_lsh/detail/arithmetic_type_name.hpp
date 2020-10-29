/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-23
 *
 * @brief Implements conversion functions from arithmetic types to their name as string representation.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARITHMETIC_TYPE_NAME_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARITHMETIC_TYPE_NAME_HPP

#include <string_view>

/**
 * @def SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME
 * @brief Defines a macro to create all possible conversion functions from arithmetic types to their name as string representation.
 * @param[in] type the data type to convert to a string
 */
#define SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME(type)                     \
template <>                                                            \
[[nodiscard]]                                                          \
inline std::string_view arithmetic_type_name<type>() { return #type; }

namespace sycl_lsh::detail {

    /**
     * @brief Tries to convert the given type to its name as string representation.
     * @details The definition is marked as **deleted** if `T` isn't an arithmetic type (without cvref-qualification).
     * @tparam T the type to convert to a string
     * @return the name of `T`
     */
    template <typename T>
    [[nodiscard]]
    inline std::string_view arithmetic_type_name() = delete;


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
    
}

#undef SYCL_LSH_CREATE_ARITHMETIC_TYPE_NAME

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARITHMETIC_TYPE_NAME_HPP
