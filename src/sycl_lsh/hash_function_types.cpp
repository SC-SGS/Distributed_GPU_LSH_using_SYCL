/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/hash_function_types.hpp"

#include "sycl_lsh/detail/utility.hpp"  // sycl_lsh::detail::to_lower_case

#include <iomanip>   // std::ios::failbit
#include <iostream>  // std::ostream, std::istream
#include <string>    // std::string

namespace sycl_lsh {

std::ostream &operator<<(std::ostream &out, const hash_function_type hash_function) {
    switch (hash_function) {
        case hash_function_type::random_projections:
            out << "random_projections";
            break;
        case hash_function_type::entropy_based:
            out << "entropy_based";
            break;
        case hash_function_type::mixed_hash_functions:
            out << "mixed_hash_functions";
            break;
    }
    return out;
}

std::istream &operator>>(std::istream &in, hash_function_type &hash_function) {
    std::string str{};
    in >> str;
    // convert string to lower case representation
    str = detail::to_lower_case(str);

    if (str == "0" || str == "random_projections" || str == "random-projections") {
        hash_function = hash_function_type::random_projections;
    } else if (str == "1" || str == "entropy_based" || str == "entropy-based" || str == "entropy") {
        hash_function = hash_function_type::entropy_based;
    } else if (str == "2" || str == "mixed_hash_functions" || str == "mixed-hash-functions" || str == "mixed") {
        hash_function = hash_function_type::mixed_hash_functions;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace sycl_lsh
