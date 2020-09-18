/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-18
 *
 * @brief The main file containing the main logic.
 */

#include <sycl_lsh/core.hpp>

#include <mpi.h>

#include <iostream>



int custom_main(int argc, char** argv) {
    sycl_lsh::communicator comm;
    sycl_lsh::errhandler handler(sycl_lsh::errhandler::type::comm);
    comm.attach_errhandler(handler);

    try {
        MPI_Comm_call_errhandler(comm.get(), 1);
    } catch (const sycl_lsh::communicator_exception& e) {
        std::cerr << "Exception thrown on rank " << e.rank() << ": " << e.what() << " (error code: " << e.error_code() << ")" << std::endl;
    }
    
    std::cout << "custom main" << std::endl;
    return 0;
}


int main(int argc, char** argv) {
    return sycl_lsh::main(argc, argv, &custom_main);
}
