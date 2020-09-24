/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-24
 *
 * @brief Implements a custom [{fmt}](https://github.com/fmtlib/fmt) formatter which can handle
 *        [`std::chrono::duration`](https://en.cppreference.com/w/cpp/chrono/duration).
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FMT_CHRONO_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FMT_CHRONO_HPP

#include <fmt/format.h>

#include <chrono>

/**
 * @brief Custom [{fmt}](https://github.com/fmtlib/fmt) formatter which can handle
 *        [`std::chrono::duration`](https://en.cppreference.com/w/cpp/chrono/duration) (i.e. it includes the respective unit (ms, s, etc.).
 */
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

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_FMT_CHRONO_HPP
