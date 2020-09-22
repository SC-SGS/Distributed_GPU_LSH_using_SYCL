/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-22
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

    // create default logger (logs to std::cout)
    sycl_lsh::mpi::logger logger(comm);

    try {

        // parse command line arguments
        sycl_lsh::argv_parser parser(argc, argv);
        // log help message if requested
        if (parser.has_argv("help")) {
            logger.log(parser.description());
            return EXIT_SUCCESS;
        }

    } catch (const std::exception& e) {
        logger.log(e.what());
//        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int main(int argc, char** argv) {
    return sycl_lsh::mpi::main(argc, argv, &custom_main);
}
