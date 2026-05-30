/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/data_set.hpp"

#include "sycl_lsh/constants.hpp"                    // sycl_lsh::real_type
#include "sycl_lsh/mpi/communicator.hpp"             // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/utility.hpp"           // SYCL_LSH_MPI_ERROR_CHECK
#include "sycl_lsh/mpi/file_parser/file_parser.hpp"  // sycl_lsh::mpi::make_file_parser
#include "sycl_lsh/mpi/logger.hpp"                   // sycl_lsh::mpi::logger

#include "sycl/sycl.hpp"  // sycl::queue

#include "mpi.h"  // MPI_Sendrecv_replace

#include <iostream>  // std::ostream

namespace sycl_lsh {

data_set::data_set(const options &opt,
                   sycl::queue &queue,
                   const mpi::communicator &comm,
                   const mpi::logger &logger) :
    queue_{ queue },
    comm_{ comm } {
    const mpi::timer mpi_timer{ comm_ };

    // parse the provided data file
    const auto parser = mpi::make_file_parser<real_type>(opt.data_file, opt.file_parser, mpi::file::mode::read, comm, logger);
    data_attributes_ = data_attributes{ parser->parse_total_size(), parser->parse_rank_size(), parser->parse_dims() };
    data_ = parser->parse_content();

    // allocate memory on the device and copy the data over
    device_ptr_ = detail::device_ptr<real_type>{ data_.shape(), queue_ };
    device_ptr_.copy_to_device(data_);

    logger.log("Created data object in {}.\n", mpi_timer.elapsed());
}

void data_set::send_receive_host_buffer() {
    const int destination = (comm_.rank() + 1) % comm_.size();
    const int source = (comm_.size() + (comm_.rank() - 1) % comm_.size()) % comm_.size();

    SYCL_LSH_MPI_ERROR_CHECK(MPI_Sendrecv_replace(data_.data(), data_.size(), mpi::detail::mpi_datatype<real_type>(), destination, 0, source, 0, comm_, MPI_STATUS_IGNORE));
}

std::ostream &operator<<(std::ostream &out, const data_set &data) {
    return out << data.get_attributes();
}

}  // namespace sycl_lsh
