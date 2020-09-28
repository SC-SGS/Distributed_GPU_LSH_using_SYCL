/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-28
 *
 * @brief Implements an enum class to determine the memory layout type: Array of Structs or Struct of Arrays.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MEMORY_LAYOUT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MEMORY_LAYOUT_HPP

#include <fmt/ostream.h>

#include <ostream>

namespace sycl_lsh {

    /**
     * @brief Enum class to determine the memory layout time.
     */
    enum class memory_layout {
        /** Array of Structs */
        aos,
        /** Structs of Array */
        soa
    };

    /**
     * @brief Print the @ref sycl_lsh::memory_layout @p layout to the output stream @p out.
     * @param[in,out] out the output stream
     * @param[in] layout the memory layout
     * @return the output stream
     */
    inline std::ostream& operator<<(std::ostream& out, const memory_layout layout) {
        switch (layout) {
            case memory_layout::aos:
                return out << "Array of Structs";
            case memory_layout::soa:
                return out << "Struct of Arrays";
        }
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MEMORY_LAYOUT_HPP
