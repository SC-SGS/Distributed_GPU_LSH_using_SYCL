/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/data_attributes.hpp"

#include "sycl_lsh/constants.hpp"  // sycl_lsh::index_type

#include "fmt/format.h"  // fmt::format

#include <iostream>  // std::ostream

namespace sycl_lsh {

data_attributes::data_attributes(const index_type total_size_p, const index_type rank_size_p, const index_type dims_p) :
    total_size{ total_size_p },
    rank_size{ rank_size_p },
    dims{ dims_p } { }

std::ostream &operator<<(std::ostream &out, const data_attributes &data_attr) {
    out << fmt::format("total_size: {}\n"
                       "rank_size: {}\n"
                       "dims: {}",
                       data_attr.total_size,
                       data_attr.rank_size,
                       data_attr.dims);
    return out;
}

}  // namespace sycl_lsh
