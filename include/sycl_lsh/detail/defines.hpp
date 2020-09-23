/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-23
 *
 * @brief Defines helper values, classes, defines, etc. used in various places.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFINES_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFINES_HPP

// defines values to test against the SYCL_LSH_TARGET
#define SYCL_LSH_TARGET_CPU 0
#define SYCL_LSH_TARGET_NVIDIA 1
#define SYCL_LSH_TARGET_AMD 2


namespace sycl_lsh::detail {

    /**
     * @brief Empty base class for the @ref sycl_lsh::options class. Only used in static_asserts.
     */
    class options_base {};

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFINES_HPP
