#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP

#include <istream>
#include <ostream>

#include <config.hpp>
#include <detail/print.hpp>
#include <entropy_based_hash_function.hpp>
#include <random_projection_hash_function.hpp>


struct hash_functions {
    static struct EntropyBased{} entropy_based;
    static struct RandomProjection{} random_projection;
};

std::ostream& operator<<(std::ostream& out, hash_functions::EntropyBased) {
    out << "entropy_based";
    return out;
}
std::ostream& operator<<(std::ostream& out, hash_functions::RandomProjection) {
    out << "random_projections";
    return out;
}


template <memory_layout layout, typename Data>
[[nodiscard]] inline auto make_hash_functions(Data& data, const MPI_Comm& communicator, hash_functions::EntropyBased) {
    return make_entropy_based_hash_functions<layout, Data>(data, communicator);
}

template <memory_layout layout, typename Data>
[[nodiscard]] inline auto make_hash_functions(Data& data, const MPI_Comm& communicator, hash_functions::RandomProjection) {
    return make_random_projections_hash_functions<layout, Data>(data, communicator);
}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP
