#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_TIMER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_TIMER_HPP

#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/type_cast.hpp>

#include <chrono>
#include <type_traits>

namespace sycl_lsh::mpi {

    class timer {
        using clock = std::chrono::steady_clock;
        using time_point = std::chrono::time_point<clock>;
    public:
        timer() : start_(clock::now()) { }
        virtual ~timer() = default;

        void restart() noexcept { start_ = clock::now(); }

        template <typename unit = std::chrono::seconds>
        auto elapsed() const {
            return std::chrono::duration_cast<unit>(clock::now() - start_);
        }

    private:
        time_point start_;
    };

    class barrier_timer {
        using clock = std::chrono::steady_clock;
        using time_point = std::chrono::time_point<clock>;
    public:
        explicit barrier_timer(const communicator& comm) : comm_(comm) {
            comm_.wait();
            start_ = clock::now();
        }

        void restart() noexcept {
            comm_.wait();
            start_ = clock::now();
        }

        template <typename unit = std::chrono::seconds>
        auto elapsed() const {
            auto dur = std::chrono::duration_cast<unit>(clock::now() - start_).count();

            decltype(dur) dur_sum = 0;
            MPI_Allreduce(&dur, &dur_sum, 1, type_cast<decltype(dur)>(), MPI_SUM, comm_.get());
            
            return unit{dur_sum / comm_.size()};
        }

    private:
        const communicator& comm_;
        time_point start_;
    };

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
