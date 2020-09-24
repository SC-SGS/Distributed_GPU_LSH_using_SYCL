/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-24
 */

#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/timer.hpp>
#include <sycl_lsh/mpi/type_cast.hpp>

#include <mpi.h>

#include <chrono>
#include <fstream>


#if defined(SYCL_LSH_BENCHMARK)
    std::ofstream sycl_lsh::mpi::timer::benchmark_out_ = std::ofstream{SYCL_LSH_BENCHMARK, std::ostream::app};
#endif


// ---------------------------------------------------------------------------------------------------------- //
//                                                constructor                                                 //
// ---------------------------------------------------------------------------------------------------------- //
sycl_lsh::mpi::timer::timer([[maybe_unused]] const communicator& comm)
#if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER || defined(SYCL_LSH_BENCHMARK)
    : comm_(comm)
#endif
{
    #if SYCL_LSH_TIMER != SYCL_LSH_NO_TIMER
        #if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
            comm_.wait();
        #endif
        start_ = clock::now();
    #endif
}


// ---------------------------------------------------------------------------------------------------------- //
//                                                   timing                                                   //
// ---------------------------------------------------------------------------------------------------------- //
void sycl_lsh::mpi::timer::restart() {
    #if SYCL_LSH_TIMER != SYCL_LSH_NO_TIMER
        #if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
            comm_.wait();
        #endif
        start_ = clock::now();
    #endif
}

template <typename unit>
[[nodiscard]]
unit sycl_lsh::mpi::timer::elapsed() const {
    #if SYCL_LSH_TIMER != SYCL_LSH_NO_TIMER
        unit dur = std::chrono::duration_cast<unit>(clock::now() - start_);

        #if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
            decltype(dur.count()) dur_sum = 0;
            MPI_Allreduce(&dur, &dur_sum, 1, type_cast<decltype(dur_sum)>(), MPI_SUM, comm_.get());
            dur = unit{dur_sum / comm_.size()};
        #endif

        #if defined(SYCL_LSH_BENCHMARK)
            if (comm_.master_rank()) {
                benchmark_out_ << dur.count() << ',';
            }
        #endif

        return dur;
    #else
        return 0;
    #endif
}

// List all possible template function specializations.
template std::chrono::nanoseconds sycl_lsh::mpi::timer::elapsed() const;
template std::chrono::microseconds sycl_lsh::mpi::timer::elapsed() const;
template std::chrono::milliseconds sycl_lsh::mpi::timer::elapsed() const;
template std::chrono::seconds sycl_lsh::mpi::timer::elapsed() const;
template std::chrono::minutes sycl_lsh::mpi::timer::elapsed() const;
template std::chrono::hours sycl_lsh::mpi::timer::elapsed() const;