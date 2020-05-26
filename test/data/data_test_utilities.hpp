/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-26
 * @brief Utility functions for the data test cases.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_TEST_UTILITIES_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_TEST_UTILITIES_HPP

#include <numeric>
#include <vector>

#include <config.hpp>
#include <data.hpp>


/**
 * @brief Tests if the values in @p data are **all** equal to the given @p correct_values.
 * @tparam Data the @ref data type
 * @param[in] data the @ref data object to test against
 * @param[in] correct_values the correct values
 * @param[in] expected_size the number of points in the @p data object
 * @param[in] expected_dims the number of dimensions in the @p data object
 */
template <typename Data>
void check_values(Data& data, const std::vector<typename Data::real_type>& correct_values,
                  const std::size_t expected_size, const std::size_t expected_dims)
{
    // check size and dimension
    ASSERT_EQ(data.size, expected_size);
    ASSERT_EQ(data.dims, expected_dims);

    // check values
    auto acc = data.buffer.template get_access<sycl::access::mode::read>();
    for (std::size_t i = 0; i < correct_values.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(acc[i], correct_values[i]);
    }
}

/**
 * @brief Tests if the values retrieved by @ref data::get_linear_id(const index_type, const index_type) const are correct.
 * @tparam Data the @ref data type
 * @param[in] data the @ref data object to test against
 * @param[in] indexing the indexing information together with the expected values
 * @param[in] expected_size the number of points in the @p data object
 * @param[in] expected_dim the number of dimensions in the @p data object
 */
template <typename Data>
void check_indexing(const Data& data, const std::vector<std::array<typename Data::index_type, 3>>& indexing,
                    const std::size_t expected_size, const std::size_t expected_dim)
{
    // check size and dimension
    ASSERT_EQ(data.size, expected_size);
    ASSERT_EQ(data.dims, expected_dim);

    // check indexing
    for (const auto& idx : indexing) {
        EXPECT_EQ(data.get_linear_id(idx[0], idx[1]), idx[2]);
    }
}

/**
 * @brief Constructs a [`std::vector<type>`](https://en.cppreference.com/w/cpp/container/vector) with the size `size * dims` containing
 *        values in the range [0, size * dim) in the given @ref memory_layout.
 * @tparam layout the @ref memory_layout type
 * @tparam type the value type
 * @param[in] size the number of points in the newly constructed data object
 * @param[in] dims the number of dimensions in the newly constructed data object
 * @return the expected values given the @p size, @p dim and @ref memory_layout (`[[nodiscard]]`)
 */
template <memory_layout layout, typename type>
[[nodiscard]] std::vector<type> expected_values(const std::size_t size, const std::size_t dims) {
    std::vector<type> vec(size * dims);
    if constexpr (layout == memory_layout::aos) {
        // fill as Array of Structs
        std::iota(vec.begin(), vec.end(), 0.0);
    } else {
        // fill as Struct of Arrays
        type val = 0.0;
        for (std::size_t point = 0; point < size; ++point) {
            for (std::size_t d = 0; d < dims; ++d) {
                vec[d * size + point] = val++;
            }
        }
    }
    return vec;
}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DATA_TEST_UTILITIES_HPP
