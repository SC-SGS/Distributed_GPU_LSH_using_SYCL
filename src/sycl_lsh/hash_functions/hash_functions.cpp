/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-30
 */

#include <sycl_lsh/hash_functions/hash_functions.hpp>

#include <ostream>


std::ostream& sycl_lsh::operator<<(std::ostream& out, const sycl_lsh::hash_functions_type type) {
    switch (type) {
        case hash_functions_type::random_projections:
            return out << "random_projections";
        case hash_functions_type::entropy_based:
            return out << "entropy_based";
    }
}