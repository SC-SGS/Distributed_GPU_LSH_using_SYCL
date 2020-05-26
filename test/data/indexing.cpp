/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-26
 *
 * @brief Test cases for the data::get_linear_id() member function.
 * @details Testsuite: *DataTest*
 * | test case name | test case description                                                                                 |
 * |:---------------|:------------------------------------------------------------------------------------------------------|
 * | GetLinearIdAoS | Test the conversion from a two-dimensional index to a flat one-dimensional index with **AoS** layout. |
 * | GetLinearIdSoA | Test the conversion from a two-dimensional index to a flat one-dimensional index with **SoA** layout. |
 */

#include <gtest/gtest.h>

#include <data.hpp>
#include "data_tests_utilities.hpp"


TEST(DataTest, GetLinearIdAoS) {
    const std::size_t size = 5;
    const std::size_t dim = 3;

    // create data object
    options opt;
    auto data = make_data<memory_layout::aos>(opt, size, dim);

    // check get_linear_id() function
    check_indexing(data, { {0, 0, 0}, {0, 2, 2}, {1, 1, 4}, {4, 2, 14} }, size, dim);
}

TEST(DataTest, GetLinearIdSoA) {
    const std::size_t size = 5;
    const std::size_t dim = 3;

    // create data object
    options opt;
    auto data = make_data<memory_layout::soa>(opt, size, dim);

    // check get_linear_id() function
    check_indexing(data, { {0, 0, 0}, {0, 2, 10}, {1, 1, 6}, {4, 2, 14} }, size, dim);
}