/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/memory_layout.hpp"

#include "sycl_lsh/detail/utility.hpp"  // sycl_lhs::detail::to_lower_case

#include <iostream>     // std::ostream, std::istream
#include <string>       // std::string
#include <string_view>  // std::string_view

namespace sycl_lsh {

std::ostream &operator<<(std::ostream &out, const memory_layout layout) {
    switch (layout) {
        case memory_layout::aos:
            return out << "aos";
        case memory_layout::soa:
            return out << "soa";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, memory_layout &layout) {
    std::string str;
    in >> str;
    str = detail::to_lower_case(str);

    if (str == "aos" || str == "array-of-structs") {
        layout = memory_layout::aos;
    } else if (str == "soa" || str == "struct-of-arrays") {
        layout = memory_layout::soa;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

std::string_view layout_type_to_full_string(const memory_layout layout) {
    switch (layout) {
        case memory_layout::aos:
            return "Array-of-Structs";
        case memory_layout::soa:
            return "Struct-of-Arrays";
    }
    return "unknown";
}

}  // namespace sycl_lsh