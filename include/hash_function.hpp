/**
 * @file
 * @author Marcel Breyer
 * @date 2020-07-29
 *
 * @brief Implements the factory functions for the hash functions classes.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP

#include <istream>
#include <ostream>

#include <config.hpp>
#include <detail/print.hpp>
#include <entropy_based_hash_function.hpp>
#include <random_projection_hash_function.hpp>


/**
 * @brief Struct to encapsulate all possible hash functions type tags.
 */
struct hash_functions {
    /**
     * @brief Tag for the entropy based hash functions.
     */
    static struct EntropyBased{} entropy_based;
    /**
     * @brief Tag for the random projection hash functions.
     */
    static struct RandomProjection{} random_projection;
};

/**
 * @brief Stream insertion operator overload for the @ref hash_functions::EntropyBased tag.
 * @param[in] out the output stream
 * @return `out`
 */
std::ostream& operator<<(std::ostream& out, hash_functions::EntropyBased) {
    out << "entropy_based";
    return out;
}
/**
 * @brief Stream insertion operator overload for the @ref hash_functions::RandomProjection tag.
 * @param[in] out the output stream
 * @return `out`
 */
std::ostream& operator<<(std::ostream& out, hash_functions::RandomProjection) {
    out << "random_projections";
    return out;
}


/**
 * @brief Constructs the entropy based hash functions.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Data represents the used data
 * @param[in] data the used data set
 * @param[in] communicator the used MPI_Comm communicator
 * @return the entropy based hash functions
 */
template <memory_layout layout, typename Data>
[[nodiscard]] inline auto make_hash_functions(Data& data, const MPI_Comm& communicator, hash_functions::EntropyBased) {
    return make_entropy_based_hash_functions<layout, Data>(data, communicator);
}

/**
 * @brief Constructs the random projection hash functions.
 * @tparam layout determines whether the hash functions are saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Data represents the used data
 * @param[in] data the used data set
 * @param[in] communicator the used MPI_Comm communicator
 * @return the random projection hash functions
 */
template <memory_layout layout, typename Data>
[[nodiscard]] inline auto make_hash_functions(Data& data, const MPI_Comm& communicator, hash_functions::RandomProjection) {
    return make_random_projection_hash_functions<layout, Data>(data, communicator);
}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_HASH_FUNCTION_HPP
