/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-21
 *
 * @brief Implements a simple MPI aware logger class.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_LOGGER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_LOGGER_HPP

#include <sycl_lsh/mpi/communicator.hpp>

#include <iostream>
#include <string_view>

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
         * @param[in] comm the used @ref sycl_lsh::communicator
         * @param[in,out] out the output-stream to log on
         */
        logger(const communicator& comm, std::ostream& out = std::cout);


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                  logging                                                   //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Log the given message @p msg **only** on the specified MPI rank @p comm_rank.
         * @param[in] comm_rank the MPI rank to log on
         * @param[in] msg the message to log
         */
        void log(const int comm_rank, const std::string_view msg);
        /**
         * @brief Log the given message @p msg **only** on the MPI rank 0 (master rank).
         * @param[in] msg the message to log
         */
        void log(const std::string_view msg);
        /**
         * @brief Log the given message @p msg **only** in the MPI rank 0 (master rank).
         * @param[in] msg the message to log
         */
        void log_on_master(const std::string_view msg);
        /**
         * @brief Log the given messages @p msg on all MPI ranks.
         * @details Gathers all messages on MPI rank 0 (master rank) to enable a deterministic output.
         * @param[in] msg the message to log
         */
        void log_on_all(const std::string_view msg);

    private:
        const communicator& comm_;
        std::ostream& out_;
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_LOGGER_HPP
