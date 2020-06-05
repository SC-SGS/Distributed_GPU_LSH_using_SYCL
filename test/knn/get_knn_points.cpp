/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-02
 *
 * @brief Test cases for the @ref knn::get_knn_points(const index_type) and @ref knn::get_knn_points(const index_type) member functions of
 *        the @ref knn class.
 * @details Testsuite: *KnnTest*
 * | test case name | test case description       |
 * |:---------------|:----------------------------|
 */

#include <vector>

#include <gtest/gtest.h>

#include <data.hpp>
#include <knn.hpp>
#include <options.hpp>


template <memory_layout layout, typename Knn, typename Data, typename Indexing>
void check_values(Knn& knns, Data& data, const std::size_t size, const std::size_t dims, const std::size_t k, Indexing idx) {
    auto acc_data = data.buffer.template get_access<sycl::access::mode::read>();

    for (std::size_t point = 0; point < size; point += 2) {
        auto data_points_1 = knns.template get_knn_points<layout>(point);
        ASSERT_EQ(data_points_1.size(), k * dims);
        for (std::size_t nn = 0; nn < k; ++nn) {
            for (std::size_t dim = 0; dim < dims; ++dim) {
                EXPECT_EQ(data_points_1[idx(nn, dim)], acc_data[data.get_linear_id(nn, dim)]);
            }
        }

        auto data_points_2 = knns.template get_knn_points<layout>(point + 1);
        ASSERT_EQ(data_points_2.size(), k * dims);
        for (std::size_t nn = 0; nn < k; ++nn) {
            for (std::size_t dim = 0; dim < dims; ++dim) {
                EXPECT_EQ(data_points_2[idx(nn, dim)], acc_data[data.get_linear_id(nn + k, dim)]);
            }
        }
    }
}


TEST(KnnTest, GetKnnPointsFromAoSAsAoS) {
    options opt;
    const std::size_t size = 10;
    const std::size_t dims = 3;
    auto data = make_data<memory_layout::aos>(opt, size, dims);
    const std::size_t k = 5;

    // construct knn objects and set dummy values
    auto knn = make_knn<memory_layout::aos>(k, data);
    auto acc = knn.buffer.template get_access<sycl::access::mode::discard_write>();
    for (std::size_t i = 0; i < knn.buffer.get_count(); ++i) {
        acc[i] = i % size;
    }

    const auto indexing = [=](const auto point, const auto nn) { return point * k + nn; };
    check_values<memory_layout::aos>(knn, data, size, dims, k, indexing);
}

TEST(KnnTest, GetKnnPointsFromAoSAsSoA) {
    options opt;
    const std::size_t size = 10;
    const std::size_t dims = 3;
    auto data = make_data<memory_layout::aos>(opt, size, dims);
    const std::size_t k = 5;

    // construct knn objects and set dummy values
    auto knn = make_knn<memory_layout::soa>(k, data);
    auto acc = knn.buffer.template get_access<sycl::access::mode::discard_write>();
    std::size_t i = 0;
    for (std::size_t i = 0; i < knn.buffer.get_count(); ++i) {
        acc[i] = i % size;
    }

    auto acc_data = data.buffer.template get_access<sycl::access::mode::read>();
    for (std::size_t i = 0; i < data.buffer.get_count(); ++i) {
        std::cout << acc_data[i] << " ";
    }
    std::cout << std::endl;
    auto acc_knn = knn.buffer.template get_access<sycl::access::mode::read>();
    for (std::size_t i = 0; i < knn.buffer.get_count(); ++i) {
        std::cout << acc_knn[i] << " ";
    }
    std::cout << std::endl;

    // check for correct values
    const auto indexing = [=](const auto point, const auto nn) { return nn * size + point; };
    check_values<memory_layout::soa>(knn, data, size, dims, k, indexing);
}