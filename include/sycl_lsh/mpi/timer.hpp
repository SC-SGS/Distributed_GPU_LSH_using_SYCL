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

        void start() {
            start_ = clock::now();
        }
        void stop() {
            end_ = clock::now();
        }
        void stop(const communicator& comm) {
            // wait for all MPI processes via a barrier
            comm.wait();
            end_ = clock::now();
        }

        template <typename unit = std::chrono::seconds>
        auto elapsed_time() const {
            return std::chrono::duration_cast<unit>(end_ - start_);
        }

        template <typename unit = std::chrono::seconds>
        auto elapsed_time(const communicator& comm) const {
            auto dur = std::chrono::duration_cast<unit>(end_ - start_).count();

            decltype(dur) dur_sum = 0;
            MPI_Reduce(&dur, &dur_sum, 1, type_cast<decltype(dur)>(), MPI_SUM, 0, comm.get());

            return unit{dur / comm.size()};
        }

    private:
        time_point start_;
        time_point end_;
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
