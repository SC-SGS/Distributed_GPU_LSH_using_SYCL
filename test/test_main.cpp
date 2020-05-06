#include <gtest/gtest.h>


// based on https://github.com/LLNL/gtest-mpi-listener //

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests, then clean up and exit. RUN_ALL_TESTS() returns 0 if all tests
    // pass and 1 if some test fails.
    return RUN_ALL_TESTS();
}