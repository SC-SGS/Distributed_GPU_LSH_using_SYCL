/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-15
 * @brief Test cases for the @ref options class.
 * @details Testsuite: *OptionsTest*
 * | test case name                 | test case description                                                 |
 * |:-------------------------------|:----------------------------------------------------------------------|
 * | DefaultConstruct               | Test the default construction of a @ref options object.               |
 * | DefaultConstructDifferentTypes | Test the construction of a @ref options object using different types. |
 * | SaveToFile                     | Test the saving of a @ref options object to a file.                   |
 */

#include <type_traits>

#include <gtest/gtest.h>

#include <options.hpp>


TEST(OptionsTest, DefaultConstruct) {
    // default construct options object
    options opt;

    // check whether default values are set
    // TODO 2020-05-07 13:35 marcel: change after meaningful default values are provided
    EXPECT_EQ(opt.k, 6);
    EXPECT_EQ(opt.num_hash_tables, 2);
    EXPECT_EQ(opt.hash_table_size, 105613);
    EXPECT_EQ(opt.num_hash_functions, 4);
    EXPECT_FLOAT_EQ(opt.w, 1.0);
}

TEST(OptionsTest, DefaultConstructDifferentTypes) {
    // create options object with different types
    options<double, int, short> opt;

    // check whether the types match
    EXPECT_TRUE( (std::is_same_v<typename decltype(opt)::real_type, double>) );
    EXPECT_TRUE( (std::is_same_v<typename decltype(opt)::index_type, int>) );
    EXPECT_TRUE( (std::is_same_v<typename decltype(opt)::hash_value_type, short>) );
}

TEST(OptionsTest, SaveToFile) {
    // create options object and set values
    options opt = options<>::factory()
            .set_num_hash_tables(4)
            .set_hash_table_size(7)
            .set_num_hash_functions(8)
            .set_w(3.1415);

    // save options
    opt.save("../../../test/options/saved_options.txt");

    // load previously saved options
    options opt_2 = options<>::factory("../../../test/options/saved_options.txt");

    // check whether values are set
    EXPECT_EQ(opt.num_hash_tables, 4);
    EXPECT_EQ(opt.hash_table_size, 7);
    EXPECT_EQ(opt.num_hash_functions, 8);
    EXPECT_FLOAT_EQ(opt.w, 3.1415);
}