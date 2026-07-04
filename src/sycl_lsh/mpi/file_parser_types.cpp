/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/mpi/file_parser_types.hpp"

#include "sycl_lsh/detail/utility.hpp"  // sycl_lsh::detail::to_lower_case

#include <iomanip>   // std::ios::failbit
#include <iostream>  // std::ostream, std::istream
#include <string>    // std::string

namespace sycl_lsh::mpi {

std::ostream &operator<<(std::ostream &out, const file_parser_type parser) {
    switch (parser) {
        case file_parser_type::binary:
            return out << "binary";
        case file_parser_type::arff:
            return out << "arff";
    }
    return out;
}

std::istream &operator>>(std::istream &in, file_parser_type &parser) {
    std::string str{};
    in >> str;
    // convert string to lower case representation
    str = detail::to_lower_case(str);

    if (str == "0" || str == "binary") {
        parser = file_parser_type::binary;
    } else if (str == "1" || str == "arff") {
        parser = file_parser_type::arff;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace sycl_lsh::mpi
