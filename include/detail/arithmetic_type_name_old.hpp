/**
 * @file
 * @author Marcel Breyer
 * @date 2020-08-31
 *
 * @brief Implements type name like functionality for arithmetic types (without cv-qualifiers).
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARITHMETIC_TYPE_NAME_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARITHMETIC_TYPE_NAME_HPP


/**
 * @def CREATE_ARITHMETIC_TYPE_NAME
 * @brief Defines a macro to create all possible conversion functions from @p type to its string name representation.
 * @param[in] type the data type to convert to its string name representation
 */
#define CREATE_ARITHMETIC_TYPE_NAME(type)                                  \
template <>                                                                \
[[nodiscard]]                                                              \
inline const char* arithmetic_type_name<type>() noexcept { return #type; }

namespace detail {

    /**
     * @brief Tries to convert the given type to its string name representation.
     * @details The definition is marked as **deleted** if `T` isn't a non cv-qualified arithmetic type.
     * @tparam T the type to convert to its string representation
     * @return the string representation
     */
    template <typename T>
    [[nodiscard]]
    inline const char* arithmetic_type_name() noexcept = delete;

    CREATE_ARITHMETIC_TYPE_NAME(bool)

    // character types
    CREATE_ARITHMETIC_TYPE_NAME(char)
    CREATE_ARITHMETIC_TYPE_NAME(char16_t)
    CREATE_ARITHMETIC_TYPE_NAME(char32_t)
    CREATE_ARITHMETIC_TYPE_NAME(wchar_t)

    // integer types
    CREATE_ARITHMETIC_TYPE_NAME(short)
    CREATE_ARITHMETIC_TYPE_NAME(unsigned short)
    CREATE_ARITHMETIC_TYPE_NAME(int)
    CREATE_ARITHMETIC_TYPE_NAME(unsigned int)
    CREATE_ARITHMETIC_TYPE_NAME(long)
    CREATE_ARITHMETIC_TYPE_NAME(unsigned long)
    CREATE_ARITHMETIC_TYPE_NAME(long long)
    CREATE_ARITHMETIC_TYPE_NAME(unsigned long long)
    CREATE_ARITHMETIC_TYPE_NAME(unsigned char)

    // floating point types
    CREATE_ARITHMETIC_TYPE_NAME(float)
    CREATE_ARITHMETIC_TYPE_NAME(double)
    CREATE_ARITHMETIC_TYPE_NAME(long double)

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARITHMETIC_TYPE_NAME_HPP