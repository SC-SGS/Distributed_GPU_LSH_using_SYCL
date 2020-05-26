/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-26
 *
 * @brief Test cases for the @ref argv_parser::has_argv(T&&) const member function of the @ref argv_parser object.
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


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwrite-strings"


TEST(ArgvParserTest, HasArgv) {
    // create command line arguments
    using namespace std::string_literals;
    std::vector<char*> argvs = {
            "a.out",
            "--data", "some path",
            "--k", "6",
            "--save_knn", "path to save knn to",
            "--num_hash_tables", "8",
            "--w", "3.1415" };

    // create command line argument parser
    argv_parser parser(argvs.size(), argvs.data());

    // check has_argv function
    EXPECT_FALSE(parser.has_argv("help"s));
    EXPECT_FALSE(parser.has_argv("options"s));
    EXPECT_FALSE(parser.has_argv("save_options"s));
    EXPECT_TRUE(parser.has_argv("data"s));
    EXPECT_TRUE(parser.has_argv("k"s));
    EXPECT_TRUE(parser.has_argv("save_knn"));
    EXPECT_TRUE(parser.has_argv("num_hash_tables"));
    EXPECT_FALSE(parser.has_argv("hash_table_size"));
    EXPECT_FALSE(parser.has_argv("num_hash_functions"));
    EXPECT_TRUE(parser.has_argv("w"));
}

TEST(ArgvParserTest, HasArgvIllegalKey) {
    // create command line arguments
    std::vector<char*> argvs = { "a.out", "--data", "some path", "--k", "6" };

    // create command line argument parser
    argv_parser parser(argvs.size(), argvs.data());

    // query an illegal key
    [[maybe_unused]] bool val;
    EXPECT_DEATH(val = parser.has_argv("foo"), "");
}


#pragma GCC diagnostic pop