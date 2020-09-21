/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-21
 */

#include <sycl_lsh/mpi/main.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <mpi.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>

namespace {
    /*
     * @brief Enum class for the different levels of thread support provided by MPI.
     * @details The values are monotonic: *single < funneled < serialized < multiple*.
     */
    enum class thread_support {
        /** only one thread will execute */
        single = MPI_THREAD_SINGLE,
        /** the process may be multi-threaded, but the application must ensure that only the main thread makes MPI calls */
        funneled = MPI_THREAD_FUNNELED,
        /** the process may be multi-threaded, and multiple threads may make MPI calls, but only one at a time */
        serialized = MPI_THREAD_SERIALIZED,
        /** multiple threads may make MPI calls, with no restrictions */
        multiple = MPI_THREAD_MULTIPLE,
    };

    /*
     * @brief Stream-insertion operator overload for the @ref thread_support enum class.
     * @param[in,out] out an output stream
     * @param[in] ts the enum class value
     * @return the output stream
     */
    inline std::ostream& operator<<(std::ostream& out, const thread_support ts) {
        switch (ts) {
            case thread_support::single:
                out << "MPI_THREAD_SINGLE";
                break;
            case thread_support::funneled:
                out << "MPI_THREAD_FUNNELED";
                break;
            case thread_support::serialized:
                out << "MPI_THREAD_SERIALIZED";
                break;
            case thread_support::multiple:
                out << "MPI_THREAD_MULTIPLE";
                break;
        }
        return out;
    }
}


int sycl_lsh::mpi::main(int argc, char** argv, sycl_lsh::mpi::custom_main_ptr func) {
    int return_code = EXIT_SUCCESS;

    // initialize the MPI environment with thread support
    constexpr thread_support required = thread_support::serialized;
    thread_support provided;
    MPI_Init_thread(&argc, &argv, static_cast<int>(required), reinterpret_cast<int*>(&provided));

    // get MPI rank
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if (provided < required) {
        // required level of thread support couldn't be provided -> exit program
        if (comm_rank == 0) {
            fmt::print(stderr,
                    "Couldn't provide the required level of thread support!\n"
                    "required: {}\n"
                    "provided: {}\n",
                    required, provided);
        }
        return_code = EXIT_FAILURE;
    } else {
        return_code = std::invoke(func, argc, argv);
    }

    // finalize the MPI environment
    MPI_Finalize();

    return return_code;
}