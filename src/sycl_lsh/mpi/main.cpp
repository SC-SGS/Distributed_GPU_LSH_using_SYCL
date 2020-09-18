/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-18
 */

#include <sycl_lsh/mpi/main.hpp>

#include <mpi.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>


int sycl_lsh::main(int argc, char** argv, sycl_lsh::custom_main_ptr func) {
    int return_code = EXIT_SUCCESS;

    // initialize the MPI environment with thread support
    constexpr int required = MPI_THREAD_SERIALIZED;
    int provided;
    MPI_Init_thread(&argc, &argv, required, &provided);

    // conversion function
    const auto level_of_thread_support_to_string = [](const int ts) -> std::string {
        switch (ts) {
            case MPI_THREAD_SINGLE:     return "MPI_THREAD_SINGLE";
            case MPI_THREAD_FUNNELED:   return "MPI_THREAD_FUNNELED";
            case MPI_THREAD_SERIALIZED: return "MPI_THREAD_SERIALIZED";
            case MPI_THREAD_MULTIPLE:   return "MPI_THREAD_MULTIPLE";
            default: return "";
        }
    };

    // get MPI rank
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if (provided < required) {
        // required level of thread support couldn't be provided -> exit program
        if (comm_rank == 0) {
            std::cerr << "Couldn't provide the required level of thread support!\n"
                      << "required: " << level_of_thread_support_to_string(required) << '\n'
                      << "provided: " << level_of_thread_support_to_string(provided) << std::endl;
        }
        return_code = EXIT_FAILURE;
    } else {
        // required level of thread support could be provided -> call custom main function
        if (comm_rank == 0) {
            std::cout << "Provided level of thread support is: " << level_of_thread_support_to_string(provided) << std::endl;
        }

        return_code = std::invoke(func, argc, argv);
    }

    // finalize the MPI environment
    MPI_Finalize();

    return return_code;
}