/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/profiling_types.hpp"

#include "sycl_lsh/detail/utility.hpp"  // sycl_lsh::detail::to_lower_case

#include <iomanip>   // std::ios::failbit
#include <iostream>  // std::ostream, std::istream
#include <string>    // std::string

namespace sycl_lsh {

std::ostream &operator<<(std::ostream &out, const profiling_types profiling_type) {
    switch (profiling_type) {
        case profiling_types::none:
            out << "none";
            break;
        case profiling_types::runtimes:
            out << "runtimes";
            break;
        case profiling_types::hws:
            out << "hws";
            break;
    }
    return out;
}

std::istream &operator>>(std::istream &in, profiling_types &profiling_type) {
    std::string str{};
    in >> str;
    // convert string to lower case representation
    str = detail::to_lower_case(str);

    if (str == "0" || str == "none") {
        profiling_type = profiling_types::none;
    } else if (str == "1" || str == "runtimes" || str == "runtime") {
        profiling_type = profiling_types::runtimes;
    } else if (str == "2" || str == "hws") {
        profiling_type = profiling_types::hws;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace sycl_lsh
