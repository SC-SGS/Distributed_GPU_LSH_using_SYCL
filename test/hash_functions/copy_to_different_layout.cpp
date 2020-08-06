/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-12
 *
 * @brief Test cases for the @ref hash_functions::get_as() member function.
 * @details Testsuite: *HashFunctionsTest*
 * | test case name | test case description                                                                    |
 * |:---------------|:-----------------------------------------------------------------------------------------|
 * | GetAoSasSoA    | Test the conversion of a @ref hash_functions object in **AoS** layout to **SoA** layout. |
 * | GetSoAasAoS    | Test the conversion of a @ref hash_functions object in **SoA** layout to **AoS** layout. |
 */

#include <gtest/gtest.h>

#include <hash_functions/random_projection_hash_function.hpp>


template <typename HashFunctions>
void check_order(HashFunctions& hash_funcs, const std::vector<typename HashFunctions::real_type>& correct_order) {
    // check values
    auto acc = hash_funcs.buffer.template get_access<sycl::access::mode::read>();
    for (std::size_t i = 0; i < correct_order.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(acc[i], correct_order[i]);
    }
}


TEST(HashFunctionsTest, GetAoSasSoA) {
    // create hash_functions object
    options opt;
    using index_type = typename decltype(opt)::index_type;

    auto data_set = make_data<memory_layout::aos>(opt, 10, 3);
    auto hash_functions_aos = make_hash_functions<memory_layout::aos>(data_set);

    // get aos as soa
    auto hash_functions_soa = hash_functions_aos.get_as<memory_layout::soa>();

    // check for correct order of values
    std::vector<typename decltype(opt)::real_type> correct_order;
    auto acc = hash_functions_aos.buffer.template get_access<sycl::access::mode::discard_write>();
    for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            for (index_type dim = 0; dim <= data_set.dims; ++dim) {
                correct_order.push_back(
                        acc[hash_table * opt.num_hash_functions * (data_set.dims + 1) + dim * opt.num_hash_functions + hash_function]);
            }
        }
    }

    check_order(hash_functions_soa, correct_order);
}

TEST(HashFunctionsTest, GetSoAasAoS) {
    // create hash_functions object
    options opt;
    using index_type = typename decltype(opt)::index_type;

    auto data_set = make_data<memory_layout::soa>(opt, 10, 3);
    auto hash_functions_soa = make_hash_functions<memory_layout::soa>(data_set);

    // get soa as aos
    auto hash_functions_aos = hash_functions_soa.get_as<memory_layout::aos>();

    // check for correct order of values
    std::vector<typename decltype(opt)::real_type> correct_order;
    auto acc = hash_functions_aos.buffer.template get_access<sycl::access::mode::discard_write>();
    for (index_type hash_table = 0; hash_table < opt.num_hash_tables; ++hash_table) {
        for (index_type hash_function = 0; hash_function < opt.num_hash_functions; ++hash_function) {
            for (index_type dim = 0; dim <= data_set.dims; ++dim) {
                correct_order.push_back(
                        acc[hash_table * opt.num_hash_functions * (data_set.dims + 1) + hash_function * (data_set.dims + 1) + dim]);
            }
        }
    }

    check_order(hash_functions_aos, correct_order);
}