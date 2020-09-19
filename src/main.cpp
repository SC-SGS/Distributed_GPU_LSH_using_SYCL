/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-19
 *
 * @brief The main file containing the main logic.
 */

#include <sycl_lsh/core.hpp>

#include <mpi.h>

#include <iostream>



int custom_main(int argc, char** argv) {
    // create MPI communicator
    sycl_lsh::communicator comm;
    // optionally: set exception handler for the communicator
    sycl_lsh::errhandler handler(sycl_lsh::errhandler::type::comm);
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
    }
    
    return EXIT_SUCCESS;
}


int main(int argc, char** argv) {
    return sycl_lsh::main(argc, argv, &custom_main);
}
