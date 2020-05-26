/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-26
 * @brief Utility functions for all test cases.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP


#include <gtest/gtest.h>


#define EXPECT_THROW_WITH_MESSAGE(statement, exception_type, msg) \
do {                                                              \
    try {                                                         \
        statement;                                                \
        FAIL() << "Expected " #exception_type;                    \
    } catch (const exception_type& e) {                           \
        EXPECT_STREQ(e.what(), msg);                              \
    } catch(...) {                                                \
        FAIL() << "Expected " #exception_type;                    \
    }                                                             \
} while(false);                                                   \


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_UTILITY_HPP
