/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-26
 *
 * @brief Test cases for the @ref argv_parser::argv_as(U&&) const member function of the @ref argv_parser object.
 * @details Testsuite: *ArgvParserTest*
 * | test case name    | test case description                                                           |
 * |:------------------|:--------------------------------------------------------------------------------|
 * | HasArgv           | Check whether the given key has been provided as command line argument.         |
 * | HasArgvIllegalKey | Check whether the given illegal key has been provided as command line argument. |
 */

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <argv_parser.hpp>
#include "../test_utilities.hpp"


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwrite-strings"


TEST(ArgvParserTest, ArgvAs) {
    // create command line arguments
    std::vector<char*> argvs = {
            "a.out",
            "--data", "some path",
            "--k", "6",
            "--w", "3.1415" };

    // create command line argument parser
    argv_parser parser(argvs.size(), argvs.data());

    // check argv_as function
    int k = parser.argv_as<int>("k");
    EXPECT_EQ(k, 6);
    double w = parser.argv_as<double>("w");
    EXPECT_DOUBLE_EQ(w, 3.1415);
    std::string data = parser.argv_as<std::string>("data");
    EXPECT_EQ(data, std::string("some path"));
}

TEST(ArgvParserTest, ArgvAsMissingKey) {
    // create command line arguments
    std::vector<char*> argvs = { "a.out", "--data", "some path", "--k", "6" };

    // create command line argument parser
    argv_parser parser(argvs.size(), argvs.data());

    // query an illegal key
    [[maybe_unused]] int i;
    EXPECT_THROW_WITH_MESSAGE(
            i = parser.argv_as<int>("num_hash_tables"),
            std::invalid_argument,
            "The requested key 'num_hash_tables' can't be found!");
}

TEST(ArgvParserTest, ArgvAsIllegalKey) {
    // create command line arguments
    std::vector<char*> argvs = { "a.out", "--data", "some path", "--k", "6" };

    // create command line argument parser
    argv_parser parser(argvs.size(), argvs.data());

    // query an illegal key
    [[maybe_unused]] int i;
    EXPECT_DEATH(i = parser.argv_as<int>("foo"), "");
}


#pragma GCC diagnostic pop