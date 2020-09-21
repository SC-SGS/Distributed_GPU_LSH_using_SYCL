/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-21
 *
 * @brief The main file containing the main logic.
 */

#include <sycl_lsh/core.hpp>

#include <fmt/format.h>
#include <mpi.h>

#include <iostream>



int custom_main(int argc, char** argv) {
    // create MPI communicator
    sycl_lsh::mpi::communicator comm;
    // optionally: set exception handler for the communicator
    sycl_lsh::mpi::errhandler handler(sycl_lsh::mpi::errhandler::type::comm);
    comm.attach_errhandler(handler);

    try {

        // parse command line arguments
        sycl_lsh::argv_parser parser(argc, argv);
        // print help message if requested
        if (parser.has_argv("help")) {
            if (comm.rank() == 0) {
                std::cout << parser.description() << std::endl;
                return EXIT_SUCCESS;
            }
        }

    } catch (const std::exception& e) {
        if (comm.rank() == 0) {
            std::cout << e.what() << std::endl;
        }
//        return EXIT_FAILURE;
    }

    sycl_lsh::mpi::logger logger(comm);

    if (comm.rank() == 0) {
        logger.log_on_all(fmt::format("Hello, World on rank: {} (master)\n", comm.rank()));
    } else {
        logger.log_on_all(fmt::format("Hello, World on rank: {}\n", comm.rank()));
    }

    return EXIT_SUCCESS;
}


int main(int argc, char** argv) {
    return sycl_lsh::mpi::main(argc, argv, &custom_main);
}
