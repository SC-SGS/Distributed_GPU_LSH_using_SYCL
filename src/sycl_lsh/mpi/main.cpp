/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/mpi/main.hpp"

#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator::main_rank

#include "fmt/format.h"  // fmt::format
#include "mpi/mpi.h"     // MPI_THREAD_SERIALIZED, MPI_Init_thread, MPI_Comm_rank, MPI_Finalize

#include <cstdlib>     // EXIT_SUCCESS, EXIT_FAILURE
#include <functional>  // std::invoke
#include <iostream>    // std::cerr, std::endl

int sycl_lsh::mpi::main(int argc, char **argv, custom_main_ptr func) {
    int return_code = EXIT_SUCCESS;

    // initialize the MPI environment with thread support
    constexpr int required = MPI_THREAD_MULTIPLE;
    int provided{};
    MPI_Init_thread(&argc, &argv, required, &provided);

    // get MPI rank
    int comm_rank{};
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if (provided < required) {
        // required level of thread support couldn't be provided -> exit program
        if (comm_rank == communicator::main_rank()) {
            std::cerr << fmt::format(
                "Couldn't provide the required level of thread support!\n"
                "required: {}\n"
                "provided: {}\n",
                required,
                provided)
                      << std::endl;
        }
        return_code = EXIT_FAILURE;
    } else {
        // the MPI environment was successfully initialized -> call the custom main function
        return_code = std::invoke(func, argc, argv);
    }

    // finalize the MPI environment
    MPI_Finalize();

    return return_code;
}