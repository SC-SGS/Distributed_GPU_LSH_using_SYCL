/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-05
 *
 * @brief Defines a custom assertion macro with more intuitive syntax and better error message.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP

#include <cstdlib>
#include <iostream>
#include <type_traits>

#include <detail/print.hpp>
#include <detail/source_location.hpp>


namespace detail {

    /**
     * @brief
     * @tparam Args parameter pack for the placeholder types (must be arithmetic types)
     * @param[in] cond the assert condition, terminates the program if evaluated to `false`
     * @param[in] cond_str the assert condition as string for a better error message
     * @param[in] loc the location where the assertion has been triggered
     * @param[in] msg the custom assert message printed after the assertion location
     * @param[in] args the arguments to fill the `printf` like placeholders in the custom error message
     */
    template <typename... Args, std::enable_if_t<(std::is_arithmetic_v<Args> && ...), int> = 0>
    inline void check(const bool cond, const char* cond_str, const source_location& loc, const char* msg, Args&&... args) {
        // check if the condition holds
        if (!cond) {
            // print source location options
            std::cout << "Assertion '" << cond_str << "' failed!\n"
                      << "  in file '" << loc.file_name() << "'\n"
                      << "  in function '" << loc.function_name() << "'\n"
                      << "  @ line " << loc.line() << "\n\n";

            // print the additional assertion message
            print(msg, std::forward<Args>(args)...);
            std::cout << std::endl;

            // abort the program
            abort();
        }
    }

}


/**
 * @def DEBUG_ASSERT
 * @brief Defines a custom `assert()` macro.
 * @details This macro is only defined in debug builds.
 * @param[in] cond the assert condition
 * @param[in] msg the custom assert message
 * @param[in] ... varying number of parameters to fill the `printf` like placeholders in the custom assert message
 */
#ifdef NDEBUG
#define DEBUG_ASSERT(cond, msg, ...)
#else
#define DEBUG_ASSERT(cond, msg, ...) \
        detail::check(cond, #cond, detail::source_location::current(PRETTY_FUNC_NAME__), msg, __VA_ARGS__)
#endif


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP
