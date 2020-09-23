/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-23
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
    logger.log("MPI_Comm_size: {}\n\n", comm.size());

    try {

        // parse command line arguments
        sycl_lsh::argv_parser parser(argc, argv);
        // log help message if requested
        if (parser.has_argv("help")) {
            logger.log(sycl_lsh::argv_parser::description());
            return EXIT_SUCCESS;
        }

        // parse options and print
        const sycl_lsh::options<float, std::uint32_t, std::uint32_t> opt(parser, logger);
        logger.log("Used options: \n{}\n", opt);

        // optionally save generated options to file
        if (parser.has_argv("options_save_file")) {
            opt.save(comm, parser, logger);
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
