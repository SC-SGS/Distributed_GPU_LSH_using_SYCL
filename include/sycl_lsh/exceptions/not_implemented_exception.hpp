/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 *
 * @brief Custom exception for not yet implemented functions.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_NOT_IMPLEMENTED_EXCEPTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_NOT_IMPLEMENTED_EXCEPTION_HPP

#include <stdexcept>

namespace sycl_lsh {

    /**
     * @brief Exception class for not yet implemented functions.
     */
    class not_implemented : public std::logic_error {
    public:
        /**
         * @brief Constructs a new @ref sycl_lsh::not_implemented exception.
         */
        not_implemented();
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_NOT_IMPLEMENTED_EXCEPTION_HPP
