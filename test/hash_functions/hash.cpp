/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-12
 *
 * @brief Test cases for the @ref hash_functions::hash() member function.
 * @details Testsuite: *HashFunctionsTest*
 * | test case name     | test case description                |
 * |:-------------------|:-------------------------------------|
 * | CalculateHashValue | Test the calculation of hash values. |
 */

#include <gtest/gtest.h>

#include <hash_function.hpp>


TEST(HashFunctionsTest, CalculateHashValue) {
    // create hash_functions object
    options opt = options<>::factory()
            .set_num_hash_tables(1)
            .set_hash_table_size(105613)
            .set_num_hash_functions(3)
            .set_w(2.0);
    auto data_set = make_data<memory_layout::aos>(opt, 1, 3);
    auto hash_functions = make_hash_functions<memory_layout::aos>(opt, data_set);

    {
        // overwrite data_set data
        auto acc = data_set.buffer.template get_access<sycl::access::mode::discard_write>();
        for (std::size_t i = 0; i < data_set.buffer.get_count(); ++i) {
            acc[i] = i;
        }
    }
    {
        // overwrite hash_functions data
        auto acc = hash_functions.buffer.template get_access<sycl::access::mode::discard_write>();
        for (std::size_t i = 0; i < hash_functions.buffer.get_count(); ++i) {
            acc[i] = i;
        }
    }

    auto acc_data = data_set.buffer.template get_access<sycl::access::mode::read>();
    auto acc_hash_functions = hash_functions.buffer.template get_access<sycl::access::mode::read>();
    auto hash_value = hash_functions.hash(0, 0, acc_data, acc_hash_functions);

    EXPECT_EQ(hash_value, 23608);
}