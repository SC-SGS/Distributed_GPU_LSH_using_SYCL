/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-15
 * @brief Test cases for the @ref options::factory class.
 * @details Testsuite: *OptionsTest*
 * | test case name                 | test case description                                                         |
 * |:-------------------------------|:------------------------------------------------------------------------------|
 * | FactoryDefaultConstruct        | Test the default factory construction of a @ref options object.               |
 * | FactoryConstructAndSet         | Test the factory construction of a @ref options object using the setters.     |
 * | FactoryConstructFromFile       | Test the factory construction of a @ref options object from a file.           |
 * | FactoryConstructDifferentTypes | Test the factory construction of a @ref options object using different types. |
 */

#include <type_traits>

#include <gtest/gtest.h>

#include <options.hpp>


TEST(OptionsTest, FactoryDefaultConstruct) {
    // create options object using the default factory initialization
    options opt = options<>::factory();

    // check whether default values are set
    // TODO 2020-05-07 13:35 marcel: change after meaningful default values are provided
    EXPECT_EQ(opt.num_hash_tables, 2);
    EXPECT_EQ(opt.hash_table_size, 105613);
    EXPECT_EQ(opt.num_hash_functions, 4);
    EXPECT_FLOAT_EQ(opt.w, 1.0);
}

TEST(OptionsTest, FactoryConstructAndSet) {
    // create options object
    options opt = options<>::factory()
            .set_num_hash_tables(4)
            .set_hash_table_size(7)
            .set_num_hash_functions(8)
            .set_w(3.1415);

    // check whether values are set
    EXPECT_EQ(opt.num_hash_tables, 4);
    EXPECT_EQ(opt.hash_table_size, 7);
    EXPECT_EQ(opt.num_hash_functions, 8);
    EXPECT_FLOAT_EQ(opt.w, 3.1415);
}

TEST(OptionsTest, FactoryConstructFromFile) {
    // create options object from file
    options opt = options<>::factory("../../../test/options/custom_options.txt");

    // check whether values are set
    EXPECT_EQ(opt.num_hash_tables, 10);
    EXPECT_EQ(opt.hash_table_size, 13);
    EXPECT_EQ(opt.num_hash_functions, 9);
    EXPECT_FLOAT_EQ(opt.w, 1.4142);
}

TEST(OptionsTest, FactoryConstructDifferentTypes) {
    // create options object with different types
    options opt = options<double, int, short>::factory().create();

    // check whether the types match
    EXPECT_TRUE( (std::is_same_v<typename decltype(opt)::real_type, double>) );
    EXPECT_TRUE( (std::is_same_v<typename decltype(opt)::index_type, int>) );
    EXPECT_TRUE( (std::is_same_v<typename decltype(opt)::hash_value_type, short>) );
}