/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/data_set.hpp"

#include "sycl_lsh/constants.hpp"                           // sycl_lsh::real_type
#include "sycl_lsh/mpi/communicator.hpp"                    // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/file_parser/file.hpp"         // sycl_lsh::mpi::detail::file::mode
#include "sycl_lsh/mpi/detail/file_parser/file_parser.hpp"  // sycl_lsh::mpi::detail::make_file_parser
#include "sycl_lsh/mpi/detail/logging.hpp"                  // sycl_lsh::mpi::detail::log
#include "sycl_lsh/mpi/detail/timer.hpp"                    // sycl_lsh::mpi::detail::timer

#include <iostream>  // std::ostream

namespace sycl_lsh {

data_set::data_set(const options &opt,
                   const mpi::communicator &comm) {
    const mpi::detail::timer mpi_timer{ comm };

    // parse the provided data file
    const auto parser = mpi::detail::make_file_parser<real_type>(opt.data_file, opt.file_parser, mpi::detail::file::mode::read, comm);
    attributes_ = attributes{ parser->parse_total_size(), parser->parse_rank_size(), parser->parse_dims() };
    data_ptr_ = std::make_shared<aos_matrix<real_type>>(parser->parse_content());

    mpi::detail::log(comm, "Created data set in {}.\n", mpi_timer.elapsed());
}

std::ostream &operator<<(std::ostream &out, const data_set &data) {
    out << fmt::format("total_size: {}\n"
                       "rank_size: {}\n"
                       "dims: {}",
                       data.get_attributes().total_size,
                       data.get_attributes().rank_size,
                       data.get_attributes().dims);
    return out;
}

}  // namespace sycl_lsh
