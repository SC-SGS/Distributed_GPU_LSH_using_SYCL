/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-28
 *
 * @brief Test cases for the construction of a @ref knn object.
 * @details Testsuite: *KnnTest*
 * | test case name | test case description       |
 * |:---------------|:----------------------------|
 * | ConstructKnn   | Construct a new knn object. |
 */

#include <gtest/gtest.h>

#include <data.hpp>
#include <knn.hpp>
#include <options.hpp>


TEST(KnnTest, ConstructKnn) {
    options opt;
    const std::size_t size = 10;
    const std::size_t dims = 3;
    auto data = make_data<memory_layout::aos>(opt, size, dims);
    const std::size_t k = 5;

    // construct knn objects
    auto knn_aos = make_knn<memory_layout::aos>(k, data);
    auto knn_soa = make_knn<memory_layout::soa>(k, data);

    // check for correct construction
    EXPECT_EQ(knn_aos.k, k);
    EXPECT_EQ(knn_aos.buffer.get_count(), size * k);
    EXPECT_EQ(knn_soa.k, k);
    EXPECT_EQ(knn_soa.buffer.get_count(), size * k);
}