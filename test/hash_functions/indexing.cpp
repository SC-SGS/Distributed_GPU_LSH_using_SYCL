/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-08
 *
 * @brief Test cases for the hash_functions::get_linear_id() member function.
 * @details Testsuite: *HashFunctionsTest*
 * | test case name | test case description                                                                                 |
 * |:---------------|:------------------------------------------------------------------------------------------------------|
 * | GetLinearIdAoS | Test the conversion from a two-dimensional index to a flat one-dimensional index with **AoS** layout. |
 * | GetLinearIdSoA | Test the conversion from a two-dimensional index to a flat one-dimensional index with **SoA** layout. |
 */

#include <gtest/gtest.h>

#include <hash_function.hpp>


template <typename HashFunctions>
void check_indexing(const HashFunctions& data, const std::vector<std::array<typename HashFunctions::index_type, 4>>& indexing) {
    // check indexing
    for (const auto& idx : indexing) {
        EXPECT_EQ(data.get_linear_id(idx[0], idx[1], idx[2]), idx[3]);
    }
}


TEST(DataTest, GetLinearIdAoS) {
    // create hash_functions object
    options opt = options<>::factory().set_num_hash_tables(2).set_num_hash_functions(4);
    auto data_set = make_data<memory_layout::aos>(opt, 10, 3);
    auto hash_functions = make_hash_functions<memory_layout::aos>(opt, data_set);

    // overwrite hash_functions data
    auto acc = hash_functions.buffer.template get_access<sycl::access::mode::discard_write>();
    for (std::size_t i = 0; i < hash_functions.buffer.get_count(); ++i) {
        acc[i] = i;
    }

    // check get_linear_id() function
     check_indexing(hash_functions, { {0, 0, 0, 0}, {0, 2, 2, 10}, {1, 1, 3, 23}, {1, 3, 1, 29} });
}

TEST(HashFunctionsTest, GetLinearIdSoA) {
    // create hash_functions object
    options opt = options<>::factory().set_num_hash_tables(2).set_num_hash_functions(4);
    auto data_set = make_data<memory_layout::aos>(opt, 10, 3);
    auto hash_functions = make_hash_functions<memory_layout::soa>(opt, data_set);

    // overwrite hash_functions data
    auto acc = hash_functions.buffer.template get_access<sycl::access::mode::discard_write>();
    for (std::size_t i = 0; i < hash_functions.buffer.get_count(); ++i) {
        acc[i] = i;
    }

    // check get_linear_id() function
     check_indexing(hash_functions, { {0, 0, 0, 0}, {0, 2, 2, 10}, {1, 1, 3, 29}, {1, 3, 1, 23} });
}