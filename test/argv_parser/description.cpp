/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-26
 *
 * @brief Test cases for the @ref argv_parser::description() const member function of the @ref argv_parser object.
 * @details Testsuite: *ArgvParserTest*
 * | test case name  | test case description                                              |
 * |:----------------|:-------------------------------------------------------------------|
 * | ArgvDescription | Check whether the argv description message is generated correctly. |
 */

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <argv_parser.hpp>


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwrite-strings"

#include <iostream>
TEST(ArgvParserTest, ArgvDescription) {
    // create command line arguments
    std::vector<char*> argvs = { "a.out", "--data", "some path", "--k", "6" };

    // create command line argument parser
    argv_parser parser(argvs.size(), argvs.data());

    std::string description =
            "Usage: ./prog --data \"path-to-data_set\" --k \"number-of-knn\" [options]\noptions:\n"
            "   --data                path to the data file (required)\n"
            "   --hash_table_size     size of each hash table (must be a prime)\n"
            "   --help                help screen\n"
            "   --k                   the number of nearest-neighbours to search for (required)\n"
            "   --num_hash_functions  number of hash functions per hash table\n"
            "   --num_hash_tables     number of hash tables to create\n"
            "   --options             path to options file\n"
            "   --save_knn            save the calculate nearest-neighbors to path\n"
            "   --save_options        save the currently used options to path\n"
            "   --w                   constant used in the hash functions\n";

    EXPECT_STREQ(parser.description().c_str(), description.c_str());
}


#pragma GCC diagnostic pop