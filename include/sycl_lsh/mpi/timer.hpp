/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements a simple timer class that can be configured via [CMake](https://cmake.org/).
 */

#ifndef SYCL_LSH_MPI_TIMER_HPP
#define SYCL_LSH_MPI_TIMER_HPP
#pragma once

#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator

#include <chrono>
#include <fstream>

namespace sycl_lsh::mpi {

// defines values to test against the SYCL_LSH_TIMER
#define SYCL_LSH_NO_TIMER           0
#define SYCL_LSH_NON_BLOCKING_TIMER 1
#define SYCL_LSH_BLOCKING_TIMER     2

/**
 * @brief Simple timer class that can be configured via [CMake](https://cmake.org/).
 * @details The timer exhibits different behavior based on the specified *SYCL_LSH_TIMER* during [CMake](https://cmake.org/)'s
 *          configuration step:
 *          - *NONE*: no timing at all
 *          - *NON_BLOCKING*: functions will be timed, but without calls to MPI_Barrier;
 *                            the elapsed times are reported separately per MPI rank
 *          - *BLOCKING*: explicit calls to MPI_Barrier on timing start (and an implicit call AFTER timing end);
 *                        the elapsed times are averaged over all MPI ranks
 *
 *          Additionally if the [CMake](https://cmake.org/) parameter *SYCL_LSH_BENCHMARK* is set, the timings are also logged to
 *          the specified file in a machine-readable way.
 */
class timer {
    /// The used [`std::chrono`](https://en.cppreference.com/w/cpp/chrono) clock.
    using clock = std::chrono::steady_clock;
    /// The [`std::chrono::time_point`](https://en.cppreference.com/w/cpp/chrono/time_point) depending on the used clock type.
    using time_point = std::chrono::time_point<clock>;

  public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Construct a new timer.
     * @details Different behavior based on the specified *SYCL_LSH_TIMER*:
     *          - *NONE*: nothing happens
     *          - *NON_BLOCKING*: directly starts timing
     *          - *BLOCKING*: calls MPI_Barrier(), afterward starts timing
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     */
    explicit timer(const communicator &comm);

    // ---------------------------------------------------------------------------------------------------------- //
    //                                                   timing                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    /**
     * @brief Reset and restart the timing.
     * @details Different behavior based on the specified *SYCL_LSH_TIMER*:
     *          - *NONE*: nothing happens
     *          - *NON_BLOCKING*: directly starts timing
     *          - *BLOCKING*: calls MPI_Barrier(), afterward starts timing
     */
    void restart();
    /**
     * @brief Returns the elapsed time since the construction of this timer or the last call to @ref sycl_lsh::mpi::timer::restart().
     * @details Different behavior based on the specified *SYCL_LSH_TIMER*:
     *          - *NONE*: returns `0`
     *          - *NON_BLOCKING*: returns the elapsed time on the current MPI rank
     *          - *BLOCKING*: calls MPI_Barrier(), afterward returns the average elapsed time on all MPI ranks
     *
     *          Additionally if the [CMake](https://cmake.org/) parameter *SYCL_LSH_BENCHMARK* is set, the timings are also logged to
     *          the specified file in a machine-readable way.
     * @tparam unit the [`std::chrono::duration`](https://en.cppreference.com/w/cpp/chrono/duration) type to use
     * @return the elapsed time with the time @p unit ([[nodiscard]])
     */
    template <typename unit = std::chrono::milliseconds>
    [[nodiscard]] unit elapsed() const;

#if defined(SYCL_LSH_BENCHMARK)
    /**
     * @brief If benchmarking is enabled (via the [CMake](https://cmake.org/) parameter *SYCL_LSH_BENCHMARK*) returns the used
     *        [`std::ofstream`](https://en.cppreference.com/w/cpp/io/basic_ofstream) to log the timings.
     * @return the output stream ([[nodiscard]])
     */
    [[nodiscard]] static std::ofstream &benchmark_out() noexcept { return benchmark_out_; }
#endif

  private:
#if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER || defined(SYCL_LSH_BENCHMARK)
    /// The communicator used to average the runtimes.
    communicator comm_;
#endif
#if SYCL_LSH_TIMER != SYCL_LSH_NO_TIMER
    /// The current start time used to calculate the elapsed time.
    time_point start_;
#endif
#if defined(SYCL_LSH_BENCHMARK)
    /// The standard output stream to output the benchmark runtimes to.
    static std::ofstream benchmark_out_;
#endif
};

}  // namespace sycl_lsh::mpi

#endif  // SYCL_LSH_MPI_TIMER_HPP
