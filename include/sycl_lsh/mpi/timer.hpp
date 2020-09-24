#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_TIMER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_TIMER_HPP

#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/type_cast.hpp>

#include <chrono>
#include <fstream>
#include <type_traits>

namespace sycl_lsh::mpi {

    class timer {
        using clock = std::chrono::steady_clock;
        using time_point = std::chrono::time_point<clock>;
    public:
        explicit timer([[maybe_unused]] const communicator& comm)
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

        void restart() {
            #if SYCL_LSH_TIMER != SYCL_LSH_NO_TIMER
                #if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER
                    comm_.wait();
                #endif
                start_ = clock::now();
            #endif
        }

        template <typename unit = std::chrono::seconds>
        auto elapsed() const {
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

#if defined(SYCL_LSH_BENCHMARK)
    [[nodiscard]]
    static std::ofstream& benchmark_out() noexcept { return benchmark_out_; }
#endif

    private:
#if SYCL_LSH_TIMER == SYCL_LSH_BLOCKING_TIMER || defined(SYCL_LSH_BENCHMARK)
        const communicator& comm_;
#endif
#if SYCL_LSH_TIMER != SYCL_LSH_NO_TIMER
        time_point start_;
#endif
#if defined(SYCL_LSH_BENCHMARK)
        static std::ofstream benchmark_out_;
#endif
    };

#if defined(SYCL_LSH_BENCHMARK)
    std::ofstream timer::benchmark_out_ = std::ofstream{SYCL_LSH_BENCHMARK, std::ostream::app};
#endif
    
}


template<class rep, class period>
struct fmt::formatter<std::chrono::duration<rep, period>> {

    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(std::chrono::duration<rep, period> dur, FormatContext& ctx) {
        if constexpr (std::is_same_v<decltype(dur), std::chrono::nanoseconds>) {
            return format_to(ctx.out(), "{}ns", dur.count());
        } else if constexpr (std::is_same_v<decltype(dur), std::chrono::microseconds>) {
            return format_to(ctx.out(), "{}us", dur.count());
        } else if constexpr (std::is_same_v<decltype(dur), std::chrono::milliseconds>) {
            return format_to(ctx.out(), "{}ms", dur.count());
        } else if constexpr (std::is_same_v<decltype(dur), std::chrono::seconds>) {
            return format_to(ctx.out(), "{}s", dur.count());
        } else if constexpr (std::is_same_v<decltype(dur), std::chrono::minutes>) {
            return format_to(ctx.out(), "{}min", dur.count());
        } else if constexpr (std::is_same_v<decltype(dur), std::chrono::hours>) {
            return format_to(ctx.out(), "{}h", dur.count());
        } else {
            return format_to(ctx.out(), "{}", dur.count());
        }
    }
    
};

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_TIMER_HPP
