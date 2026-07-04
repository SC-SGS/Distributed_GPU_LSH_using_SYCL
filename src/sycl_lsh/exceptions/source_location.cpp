/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/exceptions/source_location.hpp"

#include "sycl_lsh/mpi/environment.hpp"  // sycl_lsh::mpi::is_active

#include "mpi.h"  // MPI_Comm_rank, MPI_COMM_WORLD

#include <cstdint>   // std::uint_least32_t
#include <optional>  // std::make_optional

namespace sycl_lsh {

source_location source_location::current(const char *file_name, const char *function_name, const int line, const int column) noexcept {
    source_location loc;

    loc.file_name_ = file_name;
    loc.function_name_ = function_name;
    loc.line_ = static_cast<std::uint_least32_t>(line);
    loc.column_ = static_cast<std::uint_least32_t>(column);

    // try getting the MPI rank wrt to MPI_COMM_WORLD
    try {
        if (mpi::is_active()) {
            // prevent excessive mpi::communicator constructor calls
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            loc.world_rank_ = std::make_optional(rank);
        }
    } catch (...) {
        // std::nullopt
    }

    return loc;
}

}  // namespace sycl_lsh