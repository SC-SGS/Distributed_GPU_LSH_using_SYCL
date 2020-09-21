/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-21
 *
 * @brief Wrapper function to automatically initialize and finalize the MPI environment correctly.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MAIN_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MAIN_HPP

namespace sycl_lsh::mpi {

    /// The type of the custom main function called inside @ref sycl_lsh::main.
    using custom_main_ptr = int(*)(int, char**);

    /**
     * @brief Initializes and finalizes the MPI environment with the required level of thread support (*MPI_THREAD_SERIALIZED*)
     *        and calls the custom main function denoted by @p func.
     * @param[in] argc the number of command line arguments
     * @param[in] argv the command line arguments
     * @param[in] func the custom main function to call
     * @return the return code of @p func or [*EXIT_FAILURE*](https://en.cppreference.com/w/cpp/utility/program/EXIT_status) if the
     *         required level of thread support couldn't be satisfied
     */
    int main(int argc, char** argv, custom_main_ptr func);

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MAIN_HPP
