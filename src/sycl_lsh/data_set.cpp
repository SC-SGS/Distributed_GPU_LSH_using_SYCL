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
#include "sycl_lsh/mpi/file_parser_types.hpp"               // sycl_lsh::mpi::file_parser_type
#include "sycl_lsh/profiler.hpp"                            // sycl_lsh::profiler

#include <iostream>  // std::ostream
#include <string>    // std::string

namespace sycl_lsh {

void data_set::init(const mpi::communicator &comm, const std::string &filename, const mpi::file_parser_type file_parser) {
    const mpi::detail::timer mpi_timer{ comm };

    // parse the provided data file
    const auto parser = mpi::detail::make_file_parser<real_type>(filename, file_parser, mpi::detail::file::mode::read, comm);
    attributes_ = attributes{ parser->parse_total_size(), parser->parse_rank_size(), parser->parse_dims() };
    data_ptr_ = std::make_shared<aos_matrix<real_type>>(parser->parse_content());

    const auto runtime = mpi_timer.elapsed();
    mpi::detail::log(comm, "Created a data set with {} total data points ({} per MPI rank) and {} dimensions from '{}' in {}.\n\n", attributes_.total_size, attributes_.rank_size, attributes_.dims, filename, runtime);

    // add entry if available
    if (profiler_ != nullptr) {
        profiler_->add_entry("data_set", "total_runtime", runtime);
        profiler_->add_entry("data_set", "parser_type", file_parser);
        profiler_->add_entry("data_set", attributes_);
    }
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
