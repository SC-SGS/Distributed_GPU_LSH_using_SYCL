/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-26
 *
 * @brief Test cases for the @ref data::get_as() member function.
 * @details Testsuite: *DataTest*
 * | test case name | test case description                                                          |
 * |:---------------|:-------------------------------------------------------------------------------|
 * | GetAoSasSoA    | Test the conversion of a @ref data object in **AoS** layout to **SoA** layout. |
 * | GetSoAasAoS    | Test the conversion of a @ref data object in **SoA** layout to **AoS** layout. |
 */

#include <gtest/gtest.h>

#include <data.hpp>
#include "data_test_utilities.hpp"


TEST(Datatest, GetAoSasSoA) {
    const std::size_t size = 10;
    const std::size_t dim = 3;

    // create data object
    options opt;
    auto data = make_data<memory_layout::aos>(opt, size, dim);

    // get aos as soa
    auto data_soa = data.get_as<memory_layout::soa>();

    // check for correct values
    check_values(data_soa, expected_values<memory_layout::soa, typename decltype(opt)::real_type>(size, dim), size, dim);
}

TEST(Datatest, GetSoAasAoS) {
    const std::size_t size = 10;
    const std::size_t dim = 3;

    // create data object
    options opt;
    auto data = make_data<memory_layout::soa>(opt, size, dim);

    // get soa as aos
    auto data_aos = data.get_as<memory_layout::aos>();

    // check for correct values
    check_values(data_aos, expected_values<memory_layout::aos, typename decltype(opt)::real_type>(size, dim), size, dim);
}