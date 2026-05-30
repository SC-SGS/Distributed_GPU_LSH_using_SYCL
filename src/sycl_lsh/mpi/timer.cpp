/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/mpi/timer.hpp"

#include "sycl_lsh/mpi/communicator.hpp"      // sycl_lsh::mpi::communicator
#include "sycl_lsh/mpi/detail/type_cast.hpp"  // sycl_lsh::mpi::detail::mpi_datatype
#include "sycl_lsh/mpi/detail/utility.hpp"    // SYCL_LSH_MPI_ERROR_CHECK

#include "mpi/mpi.h"  // MPI_Allreduce

#include <chrono>   // std::chrono::duration_cast
#include <fstream>  // std::ofstream
#include <iomanip>  // std::ios_base::app

namespace sycl_lsh::mpi {

#if defined(SYCL_LSH_BENCHMARK)
std::ofstream timer::benchmark_out_ = std::ofstream{ SYCL_LSH_BENCHMARK, std::ios_base::app };
#endif

// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
timer::timer([[maybe_unused]] const communicator &comm)
#if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER || defined(SYCL_LSH_BENCHMARK)
    :
    comm_(comm)
#endif
{
#if SYCL_LSH_TIMER != SYCL_LSH_NO_TIMER
    #if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
    comm_.barrier();
    #endif
    start_ = clock::now();
#endif
}

// ---------------------------------------------------------------------------------------------------------- //
//                                                   timing                                                   //
// ---------------------------------------------------------------------------------------------------------- //
void timer::restart() {
#if SYCL_LSH_TIMER != SYCL_LSH_NO_TIMER
    #if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
    comm_.barrier();
    #endif
    start_ = clock::now();
#endif
}

template <typename unit>
[[nodiscard]] unit timer::elapsed() const {
#if SYCL_LSH_TIMER != SYCL_LSH_NO_TIMER
    unit dur = std::chrono::duration_cast<unit>(clock::now() - start_);

    #if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
    decltype(dur.count()) dur_sum = 0;
    SYCL_LSH_MPI_ERROR_CHECK(MPI_Allreduce(&dur, &dur_sum, 1, detail::mpi_datatype<decltype(dur_sum)>(), MPI_SUM, comm_));
    dur = unit{ dur_sum / comm_.size() };
    #endif

    #if defined(SYCL_LSH_BENCHMARK)
    if (comm_.is_main_rank()) {
        benchmark_out_ << dur.count() << ',';
    }
    #endif

    return dur;
#else
    return unit{};
#endif
}

// List all possible template function specializations.
template std::chrono::nanoseconds timer::elapsed() const;
template std::chrono::microseconds timer::elapsed() const;
template std::chrono::milliseconds timer::elapsed() const;
template std::chrono::seconds timer::elapsed() const;
template std::chrono::minutes timer::elapsed() const;
template std::chrono::hours timer::elapsed() const;

}  // namespace sycl_lsh::mpi