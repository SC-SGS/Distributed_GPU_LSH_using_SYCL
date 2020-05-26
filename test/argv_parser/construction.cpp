/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-26
 *
 * @brief Test cases for the construction of a @ref argv_parser object.
 * @details Testsuite: *ArgvParserTest*
 * | test case name               | test case description                                                            |
 * |:-----------------------------|:---------------------------------------------------------------------------------|
 * | ConstructArgv                | Construct a new argv_parser with all possible legal arguments.                   |
 * | ConstructArgvIllegalArgc     | Construct a new argv_parser with an illegal number of argvs (argc) (death test). |
 * | ConstructArgvIllegalArgv     | Construct a new argv_parser with an illegal argv (`nullptr`) (death test).       |
 * | ConstructArgvIllegalStart    | Construct a new argv_parser with a key that misses its leading '--'.             |
 * | ConstructArgvIllegalValue    | Construct a new argv_parser with an illegal key.                                 |
 * | ConstructArgvDuplicated      | Construct a new argv_parser with the same key provided multiple times.           |
 * | ConstructArgvMissingRequired | Construct a new argv_parser with missing required keys (`--data` and `--k`).     |
 */

#include <vector>

#include <gtest/gtest.h>

#include <argv_parser.hpp>
#include "../test_utilities.hpp"


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwrite-strings"


TEST(ArgvParserTest, ConstructArgv) {
    // create all legal command line arguments
    std::vector<char*> argvs = {
            "a.out",
            "--help",
            "--options", "path to options",
            "--save_options", "path to save options to",
            "--data", "some path",
            "--k", "6",
            "--save_knn", "path to save knn to",
            "--num_hash_tables", "8",
            "--hash_table_size", "31",
            "--num_hash_functions", "4",
            "--w", "3.1415" };

    // create command line argument parser
    argv_parser parser(argvs.size(), argvs.data());
}

TEST(ArgvParserTest, ConstructArgvIllegalArgc) {
    EXPECT_DEATH(argv_parser(0, nullptr), "");
    EXPECT_DEATH(argv_parser(-1, nullptr), "");
}

TEST(ArgvParserTest, ConstructArgvIllegalArgv) {
    EXPECT_DEATH(argv_parser(1, nullptr), "");
}

TEST(ArgvParserTest, ConstructArgvIllegalStart) {
    // create illegal command line argument
    std::vector<char*> argvs = { "a.out", "data", "bar" };

    EXPECT_THROW_WITH_MESSAGE(
            argv_parser(argvs.size(), argvs.data()),
            std::invalid_argument,
            "All argv keys must start with '--'!: data"
    );
}

TEST(ArgvParserTest, ConstructArgvIllegalValue) {
    // create illegal command line argument
    std::vector<char*> argvs = { "a.out", "--foo", "bar" };

    EXPECT_THROW_WITH_MESSAGE(
            argv_parser(argvs.size(), argvs.data()),
            std::invalid_argument,
            "Illegal argv key!: foo"
    );
}

TEST(ArgvParserTest, ConstructArgvDuplicated) {
    // create illegal command line argument
    std::vector<char*> argvs = { "a.out", "--data", "foo", "--data", "bar" };

    EXPECT_THROW_WITH_MESSAGE(
            argv_parser(argvs.size(), argvs.data()),
            std::invalid_argument,
            "Duplicate argv keys!: data"
    );
}

TEST(ArgvParserTest, ConstructArgvMissingRequired) {
    // create illegal command line argument
    std::vector<char*> argvs_1 = { "a.out", "--k", "6" };
    std::vector<char*> argvs_2 = { "a.out", "--data", "foo" };

    EXPECT_THROW_WITH_MESSAGE(
            argv_parser(argvs_1.size(), argvs_1.data()),
            std::logic_error,
            "The required command line argument --data is missing!"
    );
    EXPECT_THROW_WITH_MESSAGE(
            argv_parser(argvs_2.size(), argvs_2.data()),
            std::logic_error,
            "The required command line argument --k is missing!"
    );
}


#pragma GCC diagnostic pop