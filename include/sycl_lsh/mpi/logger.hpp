/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements a simple MPI aware logger class.
 */

#ifndef SYCL_LSH_MPI_LOGGER_HPP
#define SYCL_LSH_MPI_LOGGER_HPP
#pragma once

#include "sycl_lsh/detail/assert.hpp"         // SYCL_LSH_ASSERT
#include "sycl_lsh/mpi/communicator.hpp"      // sycl_lsh::communicator
#include "sycl_lsh/mpi/detail/type_cast.hpp"  // sycl_lsh::mpi::detail::mpi_datatype

#include "fmt/format.h"  // fmt::format
#include "mpi.h"         // MPI_Gather, MPI_Gatherv

#include <cstddef>      // std::size_t
#include <iostream>     // std::ostream, std::cout
#include <numeric>      // std::accumulate
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::forward
#include <vector>       // std::vector

namespace sycl_lsh::mpi {

/**
 * @brief Simple MPI aware logging class.
 */
class logger {
  public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new logger for the given output-stream @p out.
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     * @param[in,out] out the output-stream to log on
     */
    explicit logger(const communicator &comm, std::ostream &out = std::cout);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                  logging                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Log the given message @p msg **only** on the specified MPI rank @p comm_rank.
     * @tparam Args the types of the placeholders
     * @param[in] comm_rank the MPI rank to log on
     * @param[in] msg the message to log
     * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
     */
    template <typename... Args>
    void log(int comm_rank, std::string_view msg, Args &&...args) const;
    /**
     * @brief Log the given message @p msg **only** on the MPI main rank.
     * @tparam Args the types of the placeholders
     * @param[in] msg the message to log
     * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
     */
    template <typename... Args>
    void log(std::string_view msg, Args &&...args) const;
    /**
     * @brief Log the given message @p msg **only** in the MPI main rank.
     * @tparam Args the types of the placeholders
     * @param[in] msg the message to log
     * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
     */
    template <typename... Args>
    void log_on_main(std::string_view msg, Args &&...args) const;
    /**
     * @brief Log the given messages @p msg on all MPI ranks.
     * @details Gathers all messages on MPI rank 0 (master rank) to enable a deterministic output.
     * @tparam Args the types of the placeholders
     * @param[in] msg the message to log
     * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
     */
    template <typename... Args>
    void log_on_all(std::string_view msg, Args &&...args) const;

  private:
    /// The MPI communicator representing the MPI ranks participating in the logging activity.
    const communicator &comm_;
    /// The output stream to output the data to.
    std::ostream &out_;
};

// ---------------------------------------------------------------------------------------------------------- //
//                                                  logging                                                   //
// ---------------------------------------------------------------------------------------------------------- //
template <typename... Args>
void logger::log(const int comm_rank, const std::string_view msg, Args &&...args) const {
    SYCL_LSH_ASSERT(0 <= comm_rank && comm_rank < comm_.size(),
                    "Illegal MPI rank! Must be greater or equal than 0 and less than comm.size().");

    // print message only on requested MPI rank
    if (comm_rank == comm_.rank()) {
        out_ << fmt::format(msg, std::forward<Args>(args)...);
    }
}

template <typename... Args>
void logger::log(const std::string_view msg, Args &&...args) const {
    // print message only on the main rank
    log(communicator::main_rank(), msg, std::forward<Args>(args)...);
}

template <typename... Args>
void logger::log_on_main(const std::string_view msg, Args &&...args) const {
    // print message only on the main rank
    log(communicator::main_rank(), msg, std::forward<Args>(args)...);
}

template <typename... Args>
void logger::log_on_all(const std::string_view msg, Args &&...args) const {
    // get the sizes of each message
    std::vector<int> sizes(comm_.size());
    const std::string msg_substituted = fmt::format(msg, std::forward<Args>(args)...);
    int msg_size = static_cast<int>(msg_substituted.size());
    MPI_Gather(&msg_size, 1, detail::mpi_datatype<typename decltype(sizes)::value_type>(), sizes.data(), 1, detail::mpi_datatype<decltype(msg_size)>(), 0, comm_.get());

    // calculate total msg size
    const int total_msg_size = std::accumulate(sizes.cbegin(), sizes.cend(), 0);

    // calculate displacements
    std::vector<int> displacements(sizes.size(), 0);
    for (std::size_t i = 1; i < displacements.size(); ++i) {
        displacements[i] = displacements[i - 1] + sizes[i - 1];
    }

    // get all messages
    std::string total_msg(total_msg_size, ' ');
    MPI_Gatherv(msg_substituted.data(), static_cast<int>(msg_substituted.size()), detail::mpi_datatype<char>(), total_msg.data(), sizes.data(), displacements.data(), detail::mpi_datatype<char>(), 0, comm_.get());

    // print total msg on master rank
    if (comm_.is_main_rank()) {
        out_ << total_msg;
    }
}

}  // namespace sycl_lsh::mpi

#endif  // SYCL_LSH_MPI_LOGGER_HPP
