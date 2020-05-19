/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-19
 *
 * @brief Implements metrics to evaluate the @ref knn search results.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_EVALUATION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_EVALUATION_HPP

#include <algorithm>
#include <cmath>
#include <istream>
#include <iostream>
#include <string>

#include <config.hpp>
#include <data.hpp>
#include <knn.hpp>


template <typename Knn, typename index_type>
[[nodiscard]] double recall(Knn& knns, std::vector<index_type>& ideal_knns) {
    const index_type size = knns.get_data().size;
    const index_type k = knns.k;
    double average_recall = 0.0;

    auto acc = knns.buffer.template get_access<sycl::access::mode::read>();
    for (index_type point = 0; point < size; ++point) {
        index_type count = 0;
        for (index_type i = 0; i < k; ++i) {
            for (index_type j = 0; j < k; ++j) {
                if (acc[knns.get_linear_id(point, i)] == ideal_knns[point * k + j]) {
                    ++count;
                    break;
                }
            }
        }
        average_recall += count / static_cast<double>(k);
    }

    return average_recall / size;
}


template <typename Knn, typename index_type, typename Data>
[[nodiscard]] double error_ratio(Knn& knns, std::vector<index_type>& ideal_knns, Data& data) {
    const index_type size = knns.get_data().size;
    const index_type dims = knns.get_data().dims;
    const index_type k = knns.k;
    double error_ratio = 0.0;

    auto acc_knns = knns.buffer.template get_access<sycl::access::mode::read>();
    auto acc_data = data.buffer.template get_access<sycl::access::mode::read>();
    for (index_type point = 0; point < size; ++point) {
        std::vector<double> dist(k, 0.0);
        for (index_type i = 0; i < k; ++i) {
            for (index_type dim = 0; dim < dims; ++dim) {
                dist[i] += (acc_data[data.get_linear_id(point, dim)] * acc_data[data.get_linear_id(acc_knns[knns.get_linear_id(point, i)], dim)])
                        * (acc_data[data.get_linear_id(point, dim)] * acc_data[data.get_linear_id(acc_knns[knns.get_linear_id(point, i)], dim)]);
            }
            dist[i] = std::sqrt(dist[i]);
        }
        std::sort(dist.begin(), dist.end(), std::greater<>());
        std::vector<double> ideal_dist(k, 0.0);
        for (index_type i = 0; i < k; ++i) {
            for (index_type dim = 0; dim < dims; ++dim) {
                ideal_dist[i] += (acc_data[data.get_linear_id(point, dim)] * ideal_knns[point * k + i])
                                 * (acc_data[data.get_linear_id(point, dim)] * ideal_knns[point * k + i]);
            }
            ideal_dist[i] = std::sqrt(ideal_dist[i]);
        }
        std::sort(ideal_dist.begin(), ideal_dist.end(), std::greater<>());
        for (index_type i = 0; i < k; ++i) {
            if (dist[i] != 0.0 && ideal_dist[i] != 0.0) {
                error_ratio += dist[i] / ideal_dist[i];
            }
        }
    }

    return error_ratio / static_cast<double>(size * k);
}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_EVALUATION_HPP
