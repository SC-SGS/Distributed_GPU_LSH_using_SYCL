/**
 * @file
 * @author Marcel Breyer
 * @date 2020-10-28
 *
 * @brief Implements a simple MPI aware logger class.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_LOGGER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_LOGGER_HPP

#include <sycl_lsh/detail/assert.hpp>
#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/type_cast.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <mpi.h>

#include <iostream>
#include <numeric>
#include <ostream>
#include <string_view>
#include <utility>
#include <vector>

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
        explicit logger(const communicator& comm, std::ostream& out = std::cout);


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                  logging                                                   //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Log the given message @p msg **only** on the specified MPI rank @p comm_rank.
         * @tparam the types of the placeholders
         * @param[in] comm_rank the MPI rank to log on
         * @param[in] msg the message to log
         * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
         */
        template <typename... Args>
        void log(int comm_rank, std::string_view msg, Args&&... args) const;
        /**
         * @brief Log the given message @p msg **only** on the MPI rank 0 (master rank).
         * @tparam the types of the placeholders
         * @param[in] msg the message to log
         * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
         */
        template <typename... Args>
        void log(std::string_view msg, Args&&... args) const;
        /**
         * @brief Log the given message @p msg **only** in the MPI rank 0 (master rank).
         * @tparam the types of the placeholders
         * @param[in] msg the message to log
         * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
         */
        template <typename... Args>
        void log_on_master(std::string_view msg, Args&&... args) const;
        /**
         * @brief Log the given messages @p msg on all MPI ranks.
         * @details Gathers all messages on MPI rank 0 (master rank) to enable a deterministic output.
         * @tparam the types of the placeholders
         * @param[in] msg the message to log
         * @param[in] args the arguments to fill the [{fmt}](https://github.com/fmtlib/fmt) placeholders
         */
        template <typename... Args>
        void log_on_all(std::string_view msg, Args&&... args) const;

    private:
        const communicator& comm_;
        std::ostream& out_;
    };




    // ---------------------------------------------------------------------------------------------------------- //
    //                                                  logging                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    template <typename... Args>
    void logger::log(const int comm_rank, const std::string_view msg, Args&&... args) const {
        SYCL_LSH_DEBUG_ASSERT(0 <= comm_rank && comm_rank < comm_.size(),
                "Illegal MPI rank! Must be greater or equal than 0 and less than comm.size().");

        // print message only on requested MPI rank
        if (comm_rank == comm_.rank()) {
            fmt::print(out_, msg, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void logger::log(const std::string_view msg, Args&&... args) const {
        // print message only in master rank (MPI rank 0)
        log(0, msg, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void logger::log_on_master(const std::string_view msg, Args&&... args) const {
        // print message only in master rank (MPI rank 0)
        log(0, msg, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void logger::log_on_all(const std::string_view msg, Args&&... args) const {
        // get the sizes of each message
        std::vector<int> sizes(comm_.size());
        const std::string msg_substituted = fmt::format(msg, std::forward<Args>(args)...);
        int msg_size = msg_substituted.size();
        MPI_Gather(&msg_size, 1, type_cast<typename decltype(sizes)::value_type>(), sizes.data(), 1, type_cast<decltype(msg_size)>(), 0, comm_.get());

        // calculate total msg size
        int total_msg_size = std::accumulate(sizes.begin(), sizes.end(), 0);

        // calculate displacements
        std::vector<int> displacements(sizes.size(), 0);
        for (std::size_t i = 1; i < displacements.size(); ++i) {
            displacements[i] = displacements[i - 1] + sizes[i - 1];
        }

        // get all messages
        std::string total_msg(total_msg_size, ' ');
        MPI_Gatherv(msg_substituted.data(), msg_substituted.size(), type_cast<char>(), total_msg.data(), sizes.data(), displacements.data(), type_cast<char>(), 0, comm_.get());

        // print total msg on master rank
        if (comm_.master_rank()) {
            fmt::print(out_, total_msg);
        }
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_LOGGER_HPP
