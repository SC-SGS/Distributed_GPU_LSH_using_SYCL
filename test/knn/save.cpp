/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-29
 *
 * @brief Test cases for the @ref knn::save(const std::string&, const MPI_Comm&) member function of the @ref knn class.
 * @details Testsuite: *KnnTest*
 * | test case name | test case description                                    |
 * |:---------------|:---------------------------------------------------------|
 * | SaveKnnAoS     | Save all knns in Array of Structs memory layout to file. |
 * | SaveKnnSoA     | Save all knns in Struct of Arrays memory layout to file. |
 */

#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include <data.hpp>
#include <detail/convert.hpp>
#include <knn.hpp>
#include <options.hpp>


template <typename type>
std::vector<type> read(const std::string& file) {
    std::ifstream in(file);
    std::vector<type> vec;

    std::string line, elem;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        while (std::getline(ss, elem, ',')) {
            vec.emplace_back(detail::convert_to<type>(elem));
        }
    }

    return vec;
}

template <typename type, typename knn_t>
void check_values(knn_t& knns, const std::vector<type>& vec, const std::size_t size, const std::size_t k) {
    auto acc = knns.buffer.template get_access<sycl::access::mode::read>();
    ASSERT_EQ(knns.buffer.get_count(), vec.size());
    for (std::size_t point = 0; point < size; ++point) {
        SCOPED_TRACE(point);
        for (std::size_t nn = 0; nn < k; ++nn) {
            SCOPED_TRACE(nn);
            EXPECT_EQ(acc[knns.get_linear_id(point, nn)], vec[point * k + nn]);
        }
    }
}



TEST(KnnTest, SaveKnnAoS) {
    options opt;
    const std::size_t size = 5;
    const std::size_t dims = 3;
    auto data = make_data<memory_layout::aos>(opt, size, dims);
    const std::size_t k = 2;

    // create knn object and set dummy values
    auto knn = make_knn<memory_layout::aos>(k, data);
    auto acc = knn.buffer.template get_access<sycl::access::mode::discard_write>();
    for (std::size_t i = 0; i < knn.buffer.get_count(); ++i) {
        acc[i] = i;
    }
    // save knn to file
    const std::string file = "../../../test/knn/saved_knn_aos.txt";
    knn.save(file);

    // read data back in
    std::vector<std::size_t> vec = read<std::size_t>(file);

    // check values
    check_values<std::size_t>(knn, vec, size, k);
}

TEST(KnnTest, SaveKnnSoA) {
    options opt;
    const std::size_t size = 5;
    const std::size_t dims = 3;
    auto data = make_data<memory_layout::aos>(opt, size, dims);
    const std::size_t k = 2;

    // create knn object and set dummy values
    auto knn = make_knn<memory_layout::soa>(k, data);
    auto acc_write = knn.buffer.template get_access<sycl::access::mode::discard_write>();
    for (std::size_t i = 0; i < knn.buffer.get_count(); ++i) {
        acc_write[i] = i;
    }
    // save knn to file
    const std::string file = "../../../test/knn/saved_knn_soa.txt";
    knn.save(file);

    // read data back in
    std::vector<std::size_t> vec = read<std::size_t>(file);

    // check values
    check_values<std::size_t>(knn, vec, size, k);
}