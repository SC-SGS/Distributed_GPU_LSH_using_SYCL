/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-12
 */

#include <sycl_lsh/mpi/logger.hpp>
#include <sycl_lsh/mpi/communicator.hpp>

#include <iostream>
#include <ostream>


sycl_lsh::mpi::logger::logger(const sycl_lsh::mpi::communicator& comm, std::ostream& out) : comm_(comm), out_(out) { }