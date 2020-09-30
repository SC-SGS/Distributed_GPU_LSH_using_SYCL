/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-30
 */

#include <sycl_lsh/hash_functions/hash_functions.hpp>

#include <ostream>


std::ostream& sycl_lsh::operator<<(std::ostream& out, sycl_lsh::hash_functions::EntropyBased) {
    return out << "entropy_based";
}
std::ostream& sycl_lsh::operator<<(std::ostream& out, sycl_lsh::hash_functions::RandomProjection) {
    return out << "random_projections";
}