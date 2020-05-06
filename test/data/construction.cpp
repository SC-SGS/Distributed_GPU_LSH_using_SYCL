/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-06
 * @brief Test cases for the construction of a @ref data object.
 * @details Testsuite: *DataTest*
 * | test case name      | test case description                                                         |
 * |:--------------------|:------------------------------------------------------------------------------|
 * | DirectConstruction  | Test the construction of a @ref data object given a size and dimension.       |
 * | FileConstructionAoS | Test the construction of a @ref data object with **AoS** layout given a file. |
 * | FileConstructionSoA | Test the construction of a @ref data object with **SoA** layout given a file. |
 */

#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <data.hpp>


template <typename Data>
void check_order(Data& data, const std::vector<typename Data::data_type>& correct_order) {
    // check size and dimension
    ASSERT_EQ(data.size, 5);
    ASSERT_EQ(data.dims, 3);

    // check values
    auto acc = data.buffer.template get_access<sycl::access::mode::read>();
    for (std::size_t i = 0; i < correct_order.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(acc[i], correct_order[i]);
    }
}


TEST(DataTest, DirectConstruction) {
    // create data object
    options opt;
    auto data = make_data<memory_layout::aos>(opt, 10, 3);

    // test dimensions (data is random so no point in testing)
    EXPECT_EQ(data.size, 10);
    EXPECT_EQ(data.dims, 3);
}

TEST(DataTest, FileConstructionAoS) {
    // create data object
    options opt;
    auto data = make_data<memory_layout::aos>(opt, "../../../test/data/data_set.txt");
    std::vector<typename decltype(opt)::real_type> correct_order(15);
    std::iota(correct_order.begin(), correct_order.end(), 1);

    // check for correct order of values
    check_order(data, correct_order);
}

TEST(DataTest, FileConstructionSoA) {
    // create data object
    options opt;
    auto data = make_data<memory_layout::soa>(opt, "../../../test/data/data_set.txt");
    std::vector<typename decltype(opt)::real_type> correct_order =
            { 1, 4, 7, 10, 13, 2, 5, 8, 11, 14, 3, 6, 9, 12, 15 };

    // check for correct order of values
    check_order(data, correct_order);
}