/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/mpi/logger.hpp"

#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator

#include <ostream>  // std::ostream

namespace sycl_lsh::mpi {

logger::logger(const communicator &comm, std::ostream &out) :
    comm_{ comm },
    out_{ out } { }

}  // namespace sycl_lsh::mpi