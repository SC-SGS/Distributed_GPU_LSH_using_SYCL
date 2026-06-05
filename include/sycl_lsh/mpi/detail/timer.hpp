/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements a simple timer class that can be configured via [CMake](https://cmake.org/).
 */

#ifndef SYCL_LSH_MPI_DETAIL_TIMER_HPP
#define SYCL_LSH_MPI_DETAIL_TIMER_HPP
#pragma once

#include "sycl_lsh/mpi/communicator.hpp"  // sycl_lsh::mpi::communicator

#include <chrono>  // std::chrono::{steady_clock, time_point, milliseconds}

namespace sycl_lsh::mpi::detail {

// defines values to test against the SYCL_LSH_TIMER
/// The macro definition representing the NON_BLOCKING timer implementation.
#define SYCL_LSH_NON_BLOCKING_TIMER 0
/// The macro definition representing the BLOCKING timer implementation.
#define SYCL_LSH_BLOCKING_TIMER 1

/**
 * @brief Simple timer class that can be configured via [CMake](https://cmake.org/).
 * @details The timer exhibits different behavior based on the specified *SYCL_LSH_TIMER* during [CMake](https://cmake.org/)'s
 *          configuration step:
 *          - *NON_BLOCKING*: functions will be timed, but without calls to @ref sycl_lsh::mpi::communicator::barrier;
 *                            the elapsed times are reported separately per MPI rank
 *          - *BLOCKING*: explicit calls to @ref sycl_lsh::mpi::communicator::barrier on timing start (and an implicit call AFTER timing end);
 *                        the elapsed times are averaged over all MPI ranks
 */
class timer {
    /// The used [std::chrono](https://en.cppreference.com/w/cpp/chrono) clock.
    using clock = std::chrono::steady_clock;
    /// The [std::chrono::time_point](https://en.cppreference.com/w/cpp/chrono/time_point) depending on the used clock type.
    using time_point = std::chrono::time_point<clock>;

  public:
    /**
     * @brief Construct a new timer.
     * @details Different behavior based on the specified *SYCL_LSH_TIMER*:
     *          - *NON_BLOCKING*: directly starts timing
     *          - *BLOCKING*: calls @ref sycl_lsh::mpi::communicator::barrier, afterward starts timing
     * @param[in] comm the used @ref sycl_lsh::mpi::communicator
     */
    explicit timer(const communicator &comm);

    /**
     * @brief Reset and restart the timing.
     * @details Different behavior based on the specified *SYCL_LSH_TIMER*:
     *          - *NON_BLOCKING*: directly starts timing
     *          - *BLOCKING*: calls @ref sycl_lsh::mpi::communicator::barrier, afterward starts timing
     */
    void restart();

    /**
     * @brief Returns the elapsed time since the construction of this timer or the last call to @ref sycl_lsh::mpi::detail::timer::restart().
     * @details Different behavior based on the specified *SYCL_LSH_TIMER*:
     *          - *NON_BLOCKING*: returns the elapsed time on the current MPI rank
     *          - *BLOCKING*: calls @ref sycl_lsh::mpi::communicator::barrier, afterward returns the average elapsed time on all MPI ranks
     * @tparam unit the [std::chrono::duration](https://en.cppreference.com/w/cpp/chrono/duration) type to use
     * @return the elapsed time with the time @p unit ([[nodiscard]])
     */
    template <typename unit = std::chrono::milliseconds>
    [[nodiscard]] unit elapsed() const;

  private:
    /// The @ref sycl_lsh::mpi::communicator used to average the runtimes if the SYCL_LSH_BLOCKING_TIMER is used.
    communicator comm_;
    /// The current start time used to calculate the elapsed time.
    time_point start_;
};

}  // namespace sycl_lsh::mpi::detail

#endif  // SYCL_LSH_MPI_DETAIL_TIMER_HPP
