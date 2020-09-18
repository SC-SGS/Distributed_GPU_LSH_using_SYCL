/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-03
 *
 * @brief Contains global constants, typedefs and enums.
 */

#ifndef DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
#define DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP

#include <CL/sycl.hpp>

#include <ostream>

namespace sycl = cl::sycl;
/// Namespace containing helper classes and functions.
namespace detail {
    /**
     * @brief Empty base class for the @ref options class. Only for static_asserts.
     */
    class options_base {};
    /**
     * @brief Empty base class for the @ref data class. Only for static_asserts.
     */
    class data_base {};
    /**
     * @brief Empty base class for the hash functions classes @ref entropy_based_hash_functions or @ref random_projection_hash_functions.
     * Only for static_asserts.
     */
    class hash_functions_base {};
    /**
     * @brief Empty base class for the @ref hash_tables class. Only for static_asserts.
     */
    class hash_tables_base {};
    /**
     * @brief Empty base class for the @ref knn class. Only for static_asserts.
     */
    class knn_base {};
}


/// The memory layout.
enum class memory_layout {
    /// Array of Structs
    aos,
    /// Struct of Arrays
    soa
};

std::ostream& operator<<(std::ostream& out, const memory_layout layout) {
    switch (layout) {
        case memory_layout::aos:
            out << "Array of Structs";
            break;
        case memory_layout::soa:
            out << "Struct of Arrays";
            break;
    }
    return out;
}

/// Shorthand macro for std::enable_if
#define REQUIRES(cond) std::enable_if_t<(cond), int> = 0


#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_CONFIG_HPP
