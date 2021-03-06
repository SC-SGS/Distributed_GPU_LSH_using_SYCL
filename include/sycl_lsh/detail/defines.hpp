/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-12
 *
 * @brief Defines helper values, classes, defines, etc. used in various places.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFINES_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFINES_HPP

// defines values to test against the SYCL_LSH_IMPLEMENTATION
#define SYCL_LSH_IMPLEMENTATION_HIPSYCL 0
#define SYCL_LSH_IMPLEMENTATION_COMPUTECPP 1
#define SYCL_LSH_IMPLEMENTATION_ONEAPI 2

// defines values to test against the SYCL_LSH_TARGET
#define SYCL_LSH_TARGET_CPU 0
#define SYCL_LSH_TARGET_NVIDIA 1
#define SYCL_LSH_TARGET_AMD 2
#define SYCL_LSH_TARGET_INTEL 3

// defines values to test against the SYCL_LSH_TIMER
#define SYCL_LSH_NO_TIMER 0
#define SYCL_LSH_NON_BLOCKING_TIMER 1
#define SYCL_LSH_BLOCKING_TIMER 2

namespace sycl_lsh::detail {

    /**
     * @brief Empty base class for the @ref sycl_lsh::options class. Only used in static_asserts.
     */
    class options_base {};
    /**
     * @brief Empty base class for the @ref sycl_lsh::data class. Only used in static_asserts.
     */
    class data_base {};
    /**
     * @brief Empty base class for the hash function classes. Only used in static_asserts.
     */
    class hash_functions_base {};
    /**
     * @brief Empty base class fot the hash table class. Only used in static_asserts.
     */
     class hash_tables_base {};
    /**
     * @brief Empty base class for the knn class. Only used in static_asserts.
     */
    class knn_base {};

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFINES_HPP
