/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-06
 *
 * @brief Test cases for the @ref data::get_linear_id() member function.
 * @details Testsuite: *DataTest*
 * | test case name | test case description                                                                                 |
 * |:---------------|:------------------------------------------------------------------------------------------------------|
 * | GetLinearIdAoS | Test the conversion from a two-dimensional index to a flat one-dimensional index with **AoS** layout. |
 * | GetLinearIdSoA | Test the conversion from a two-dimensional index to a flat one-dimensional index with **SoA** layout. |
 */

#include <array>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <data.hpp>


template <typename Data>
void check_indexing(const Data& data, const std::vector<std::array<typename Data::index_type, 3>>& indexing) {
    // check size and dimension
    ASSERT_EQ(data.size, 5);
    ASSERT_EQ(data.dims, 3);

    // check indexing
    for (const auto& idx : indexing) {
        EXPECT_EQ(data.get_linear_id(idx[0], idx[1]), idx[2]);
    }
}


TEST(DataTest, GetLinearIdAoS) {
    // create data object
    options opt;
    auto data = make_data<memory_layout::aos>(opt, "../../../test/data/data_set.txt");

    // check get_linear_id() function
    check_indexing(data, { {0, 0, 0}, {0, 2, 2}, {1, 1, 4}, {4, 2, 14} });
}

TEST(DataTest, GetLinearIdSoA) {
    // create data object
    options opt;
    auto data = make_data<memory_layout::soa>(opt, "../../../test/data/data_set.txt");

    // check get_linear_id() function
    check_indexing(data, { {0, 0, 0}, {0, 2, 10}, {1, 1, 6}, {4, 2, 14} });
}