/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements simple MPI aware logging functions.
 */

#ifndef SYCL_LSH_MPI_DETAIL_LOGGING_HPP
#define SYCL_LSH_MPI_DETAIL_LOGGING_HPP
#pragma once

#include "sycl_lsh/detail/assert.hpp"         // SYCL_LSH_ASSERT
#include "sycl_lsh/mpi/communicator.hpp"      // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/type_cast.hpp"  // sycl_lsh::mpi::detail::mpi_datatype
#include "sycl_lsh/mpi/detail/utility.hpp"    // SYCL_LSH_MPI_ERROR_CHECK

#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <iostream>     // std::clog, std::endl
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::forward
#include <vector>       // std::vector

namespace sycl_lsh::mpi::detail {

/**
 * @brief Log the given message @p msg **only** on the specified MPI rank @p comm_rank.
 * @tparam Args the types of the placeholders
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 * @param[in] comm_rank the MPI rank to log on
 * @param[in] msg the message to log
 * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
 */
template <typename... Args>
void log(const communicator &comm, const int comm_rank, const std::string_view msg, Args &&...args) {
    SYCL_LSH_ASSERT(comm_rank < comm.size(),
                    "Illegal MPI rank {}! Must be greater or equal than 0 and less than comm.size().",
                    comm_rank);

    // print message only on requested MPI rank
    if (comm_rank == comm.rank()) {
        std::clog << fmt::format(msg, std::forward<Args>(args)...);
    }
}

/**
 * @brief Log the given message @p msg **only** on the MPI main rank.
 * @tparam Args the types of the placeholders
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 * @param[in] msg the message to log
 * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
 */
template <typename... Args>
void log(const communicator &comm, const std::string_view msg, Args &&...args) {
    // print message only on the main rank
    log(comm, communicator::main_rank(), msg, std::forward<Args>(args)...);
}

/**
 * @brief Log the given message @p msg **only** in the MPI main rank.
 * @tparam Args the types of the placeholders
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 * @param[in] msg the message to log
 * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
 */
template <typename... Args>
void log_on_main(const communicator &comm, const std::string_view msg, Args &&...args) {
    // print message only on the main rank
    log(comm, communicator::main_rank(), msg, std::forward<Args>(args)...);
}

/**
 * @brief Log the given messages @p msg from all MPI ranks. Only the main rank does an actual command line output.
 * @details Gathers all messages on MPI rank 0 (main rank) to enable a deterministic output.
 * @tparam Args the types of the placeholders
 * @param[in] comm the used @ref sycl_lsh::mpi::communicator
 * @param[in] msg the message to log
 * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
 */
template <typename... Args>
void log_from_all(const communicator &comm, const std::string_view msg, Args &&...args) {
    // substitute the message
    const std::string msg_substituted = fmt::format(msg, std::forward<Args>(args)...);
    // gather the full string on the MPI main rank
    const std::vector<std::string> msg_parts = comm.gather(msg_substituted);
    // print full msg on the MPI main rank
    if (comm.is_main_rank()) {
        std::clog << fmt::format("{}\n", fmt::join(msg_parts, "")) << std::endl;
    }
}

}  // namespace sycl_lsh::mpi::detail

#endif  // SYCL_LSH_MPI_DETAIL_LOGGING_HPP
