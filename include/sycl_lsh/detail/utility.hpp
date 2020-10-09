/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-09
 *
 * @brief Different utility functions or macros.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP

#include <string_view>

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
    inline bool contains(const std::string_view str, const std::string_view substr) noexcept {
        return str.find(substr) != std::string_view::npos;
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP
