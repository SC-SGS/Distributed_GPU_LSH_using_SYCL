/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-06
 *
 * @brief Test cases for the @ref data::get_as() member function.
 * @details Testsuite: *DataTest*
 * | test case name | test case description                                                          |
 * |:---------------|:-------------------------------------------------------------------------------|
 * | GetAoSasSoA    | Test the conversion of a @ref data object in **AoS** layout to **SoA** layout. |
 * | GetSoAasAoS    | Test the conversion of a @ref data object in **SoA** layout to **AoS** layout. |
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


TEST(Datatest, GetAoSasSoA) {
    // create data object
    options opt;
    auto data = make_data<memory_layout::aos>(opt, "../../../test/data/data_set.txt");

    // get aos as soa
    auto data_soa = data.get_as<memory_layout::soa>();

    // check for correct order of values
    std::vector<typename decltype(opt)::real_type> correct_order =
        { 1, 4, 7, 10, 13, 2, 5, 8, 11, 14, 3, 6, 9, 12, 15 };
    check_order(data_soa, correct_order);
}

TEST(Datatest, GetSoAasAoS) {
    // create data object
    options opt;
    auto data = make_data<memory_layout::soa>(opt, "../../../test/data/data_set.txt");

    // get soa as aos
    auto data_soa = data.get_as<memory_layout::aos>();

    // check for correct order of values
    std::vector<typename decltype(opt)::real_type> correct_order(15);
    std::iota(correct_order.begin(), correct_order.end(), 1);
    check_order(data_soa, correct_order);
}