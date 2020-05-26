/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-26
 *
 * @brief Test cases for the construction of a @ref data object.
 * @details Testsuite: *DataTest*
 * | test case name        | test case description                                                                       |
 * |:----------------------|:--------------------------------------------------------------------------------------------|
 * | DirectConstructionAos | Test the construction of a @ref data object with **AoS** layout given a size and dimension. |
 * | DirectConstructionSoA | Test the construction of a @ref data object with **SoA** layout given a size and dimension. |
 * | FileConstructionAoS   | Test the construction of a @ref data object with **AoS** layout given a file.               |
 * | FileConstructionSoA   | Test the construction of a @ref data object with **SoA** layout given a file.               |
 */

#include <gtest/gtest.h>

#include <data.hpp>
#include "data_test_utilities.hpp"


TEST(DataTest, DirectConstructionAoS) {
    const std::size_t size = 10;
    const std::size_t dim = 3;

    // create data object
    options opt;
    auto data = make_data<memory_layout::aos>(opt, size, dim);

    // check for correct values
    check_values(data, expected_values<memory_layout::aos, typename decltype(opt)::real_type>(size, dim), size, dim);
}

TEST(DataTest, DirectConstructionSoA) {
    const std::size_t size = 10;
    const std::size_t dim = 3;

    // create data object
    options opt;
    auto data = make_data<memory_layout::soa>(opt, size, dim);

    // check for correct values
    check_values(data, expected_values<memory_layout::soa, typename decltype(opt)::real_type>(size, dim), size, dim);
}


TEST(DataTest, FileConstructionAoS) {
    const std::size_t size = 5;
    const std::size_t dim = 3;

    // create data object
    options opt;
    auto data = make_data<memory_layout::aos>(opt, "../../../test/data/data_set.txt");

    // check for correct order of values
    check_values(data, expected_values<memory_layout::aos, typename decltype(opt)::real_type>(size, dim), size, dim);
}

TEST(DataTest, FileConstructionSoA) {
    const std::size_t size = 5;
    const std::size_t dim = 3;

    // create data object
    options opt;
    auto data = make_data<memory_layout::soa>(opt, "../../../test/data/data_set.txt");

    // check for correct order of values
    check_values(data, expected_values<memory_layout::soa, typename decltype(opt)::real_type>(size, dim), size, dim);
}