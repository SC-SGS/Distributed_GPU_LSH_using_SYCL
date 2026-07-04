/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/mpi/detail/timer.hpp"

#include "sycl_lsh/mpi/communicator.hpp"      // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/type_cast.hpp"  // sycl_lsh::mpi::detail::mpi_datatype
#include "sycl_lsh/mpi/detail/utility.hpp"    // SYCL_LSH_MPI_ERROR_CHECK

#include "mpi/mpi.h"  // MPI_Allreduce

#include <chrono>  // std::chrono::duration_cast

namespace sycl_lsh::mpi::detail {

timer::timer(const communicator &comm) :
    comm_(comm) {
#if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
    comm_.barrier();
#endif
    start_ = clock::now();
}

void timer::restart() {
#if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
    comm_.barrier();
#endif
    start_ = clock::now();
}

template <typename unit>
[[nodiscard]] unit timer::elapsed() const {
    unit dur = std::chrono::duration_cast<unit>(clock::now() - start_);

#if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
    decltype(dur.count()) dur_sum = 0;
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Allreduce(&dur, &dur_sum, 1, detail::mpi_datatype<decltype(dur_sum)>(), MPI_SUM, comm_));
    dur = unit{ dur_sum / comm_.size() };
#endif

    return dur;
}

// List all possible template function specializations.
template std::chrono::nanoseconds timer::elapsed() const;
template std::chrono::microseconds timer::elapsed() const;
template std::chrono::milliseconds timer::elapsed() const;
template std::chrono::seconds timer::elapsed() const;
template std::chrono::minutes timer::elapsed() const;
template std::chrono::hours timer::elapsed() const;

}  // namespace sycl_lsh::mpi::detail
