/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-29
 *
 * @brief Test cases for the @ref knn::get_knn_ids(const index_type) and @ref knn::get_knn_points(const index_type) member functions of
 *        the @ref knn class.
 * @details Testsuite: *KnnTest*
 * | test case name | test case description                                                        |
 * |:---------------|:-----------------------------------------------------------------------------|
 * | GetKnnIdsAoS   | Check the found k-nearest-neighbors for a knn object with AoS memory layout. |
 * | GetKnnIdsSoA   | Check the found k-nearest-neighbors for a knn object with SoA memory layout. |
 */

#include <vector>

#include <gtest/gtest.h>

#include <data.hpp>
#include <knn.hpp>
#include <options.hpp>


template <typename knn_t>
void check_values(knn_t& knns, const std::size_t size, const std::size_t k) {
    // check for correct values
    std::size_t value = 0;
    for (std::size_t point = 0; point < size; ++point) {
        SCOPED_TRACE(point);
        auto ids = knns.get_knn_ids(point);
        ASSERT_EQ(ids.size(), k);
        for (std::size_t nn = 0; nn < k; ++nn) {
            EXPECT_EQ(ids[nn], value++);
        }
    }
}


TEST(KnnTest, GetKnnIdsAoS) {
    options opt;
    const std::size_t size = 10;
    const std::size_t dims = 3;
    auto data = make_data<memory_layout::aos>(opt, size, dims);
    const std::size_t k = 5;

    // construct knn objects and set dummy values
    auto knn = make_knn<memory_layout::aos>(k, data);
    auto acc = knn.buffer.template get_access<sycl::access::mode::discard_write>();
    for (std::size_t i = 0; i < knn.buffer.get_count(); ++i) {
        acc[i] = i;
    }

    // check for correct values
    check_values(knn, size, k);
}

TEST(KnnTest, GetKnnIdsSoA) {
    options opt;
    const std::size_t size = 10;
    const std::size_t dims = 3;
    auto data = make_data<memory_layout::aos>(opt, size, dims);
    const std::size_t k = 5;

    // construct knn objects and set dummy values
    auto knn = make_knn<memory_layout::soa>(k, data);
    auto acc = knn.buffer.template get_access<sycl::access::mode::discard_write>();
    std::size_t i = 0;
    for (std::size_t point = 0; point < size; ++point) {
        for (std::size_t nn = 0; nn < k; ++nn) {
            acc[knn.get_linear_id(point, nn)] = i++;
        }
    }

    // check for correct values
    check_values(knn, size, k);
}