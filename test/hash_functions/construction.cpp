/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-12
 *
 * @brief Test cases for the construction of a @ref hash_functions object.
 * @details Testsuite: *HashFunctionsTest*
 * | test case name   | test case description                                                                 |
 * |:-----------------|:--------------------------------------------------------------------------------------|
 * | DefaultConstruct | Test the construction of @ref hash_functions objects with **AoS** and **SoA** layout. |
 */

#include <gtest/gtest.h>

#include <hash_function.hpp>


TEST(HashFunctionsTest, DefaultConstruction) {
    // create hash_functions object
    options opt;
    auto data_set = make_data<memory_layout::aos>(opt, 10, 3);
    auto hash_functions_aos = make_hash_functions<memory_layout::aos>(data_set);
    auto hash_functions_soa = make_hash_functions<memory_layout::soa>(data_set);

    auto acc_aos = hash_functions_aos.buffer.template get_access<sycl::access::mode::read>();
    auto acc_soa = hash_functions_soa.buffer.template get_access<sycl::access::mode::read>();

    using index_type = typename decltype(opt)::index_type;

    // TODO 2020-05-08 14:50 marcel: uncomment if using seeded random numbers
    for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            for (index_type dim = 0; dim < data_set.dims; ++dim) {
                EXPECT_FLOAT_EQ(
                        acc_aos[hash_functions_aos.get_linear_id(hash_table, hash_function, dim)],
                        acc_soa[hash_functions_soa.get_linear_id(hash_table, hash_function, dim)]);
            }
        }
    }
}