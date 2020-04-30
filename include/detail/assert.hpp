#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <detail/print.hpp>
#include <detail/source_location.hpp>

namespace detail {

    template <typename... Args>
    inline void check(const bool cond, const char* cond_str, const source_location& loc, const char* msg, Args&&... args) {
        // check if assertion holds
        if (!cond) {
            std::cout << "Assertion '" << cond_str << "' failed!\n"
                      << "  in file '" << loc.file_name() << "'\n"
                      << "  in function '" << loc.function_name() << "'\n"
                      << "  @ line " << loc.line() << "\n\n";

            detail::print(msg, std::forward<Args>(args)...);
            std::cout << std::endl;

            abort();
        }
    }

}

#ifdef NDEBUG
#define DEBUG_ASSERT(cond, msg, ...)
#else
#define DEBUG_ASSERT(cond, msg, ...) \
        detail::check(cond, #cond, detail::source_location::current(PRETTY_FUNC_NAME__), msg, __VA_ARGS__)
#endif

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP
