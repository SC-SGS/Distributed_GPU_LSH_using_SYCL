/**
 * @file
 * @author Marcel Breyer
 * @date 2020-11-04
 *
 * @brief Implements the `hash_combine()` function for different hash value type sizes.
 */


#ifndef DISTRIBUTED_K_NEAREST_NEIGHBORS_USING_LOCALITY_SENSITIVE_HASHING_AND_SYCL_HASH_COMBINE_HPP
#define DISTRIBUTED_K_NEAREST_NEIGHBORS_USING_LOCALITY_SENSITIVE_HASHING_AND_SYCL_HASH_COMBINE_HPP

#include <cstdint>

/**
 * @def SYCL_LSH_CREATE_HASH_COMBINE
 * @brief Defines a macro to create all possible `hash_combine()` functions based on the size of the
 *        @ref sycl_lsh::options::hash_value_type size.
 * @param[in] type the hash value type
 * @param[in] const_1 the first constant (magic number)
 * @param[in] const_2 the second constant (left-shift)
 * @param[in] const_3 the third constant (right-shift)
 */
#define SYCL_LSH_CREATE_HASH_COMBINE(type, const_1, const_2, const_3)                                                               \
[[nodiscard]]                                                                                                                       \
inline type hash_combine(const type seed, const type val) noexcept {                                                                \
    return seed ^ (val + static_cast<type>(const_1) + (seed << static_cast<type>(const_2)) + (seed >> static_cast<type>(const_3))); \
}

namespace sycl_lsh::detail {

    SYCL_LSH_CREATE_HASH_COMBINE(std::uint16_t, 0x9e37U, 3, 1)
    SYCL_LSH_CREATE_HASH_COMBINE(std::uint32_t, 0x9e3779b9U, 6, 2)
    SYCL_LSH_CREATE_HASH_COMBINE(std::uint64_t, 0x9e3779b97f4a7c15LLU, 12, 4)

}

#undef SYCL_LSH_CREATE_HASH_COMBINE

#endif // DISTRIBUTED_K_NEAREST_NEIGHBORS_USING_LOCALITY_SENSITIVE_HASHING_AND_SYCL_HASH_COMBINE_HPP
