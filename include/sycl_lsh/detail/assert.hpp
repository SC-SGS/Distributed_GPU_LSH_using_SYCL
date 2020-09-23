/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-23
 *
 * @brief Defines a custom assertion macro with more intuitive syntax and better error message.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP

#include <sycl_lsh/detail/defines.hpp>

#include <mpi.h>

#include <cstdlib>
#include <stdio.h>
#include <utility>

namespace sycl_lsh::detail {

    /**
     * @brief A dummy function such that the compiler can catch potential format string errors.
     * @details This functions never gets called!
     * @param[in] format the `printf` format string
     * @param[in] ... the arguments to fill the `printf` placeholders in the formatting string
     */
    void check_args(const char* format, ...) __attribute__ ((format (printf, 1, 2)));

    /**
     * @brief Custom assertion function called in the `SYCL_LSH_DEBUG_ASSERT` and `SYCL_LSH_DEBUG0_ASSERT` macros.
     * @details If the assert condition @p cond evaluates to `false`, the condition, location and custom message are printed on the
     *          [`stderr stream`](https://en.cppreference.com/w/cpp/io/c/std_streams). Afterwards the program terminates with a call to
     *          `MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE)` if and only if the current `SYCL_LSH_TARGET` equals `SYCL_LSH_TARGET_CPU`.
     * @tparam Args the types of the placeholders
     * @param[in] file the name of the file in which the assertion occurred
     * @param[in] function the name of the function in which the assertion occurred
     * @param[in] line the line in which the assertion occurred
     * @param[in] cond the assert condition, terminates the program if evaluated to `false`
     * @param[in] cond_str the assert condition as string for a better error message
     * @param[in] msg the custom assert message printed after the assertion location
     * @param[in] args the arguments to fill the `printf` placeholders in the custom error message
     */
    template <typename... Args>
    inline void assert_function(const char* file, const char* function, const int line,
                                const bool cond, const char* cond_str, const char* msg, Args&&... args)
    {
        // check if the assertion holds
        if (!cond) {
            // create static sized buffer
            constexpr int buffer_size = 2048;
            char buffer[buffer_size];
            int wrote_size;

            // write source location message to buffer
            wrote_size = snprintf(buffer, buffer_size, "Assertion '%s' failed!\n"
                   "  in file '%s'\n"
                   "  in function '%s'\n"
                   "  @ line %i\n\n",
                   cond_str, file, function, line);

            // write assertion message to buffer if space is left
            if (wrote_size >= 0 && wrote_size < buffer_size) {
                snprintf(buffer + wrote_size, buffer_size - wrote_size, msg, std::forward<Args>(args)...);
            }

            // print buffer to stderr
            fprintf(stderr, "%s\n", buffer);

            // if the current target is CPU, abort the program after a failed assertion
#if SYCL_LSH_TARGET == SYCL_LSH_TARGET_CPU
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
        }
    }

}

/**
 * @def SYCL_LSH_PRETTY_FUNC_NAME__
 * @brief The @ref SYCL_LSH_PRETTY_FUNC_NAME__ macro is defined as `__PRETTY_FUNC__` ([*GCC*](https://gcc.gnu.org/) and
 * [*clang*](https://clang.llvm.org/)), `__FUNCSIG__` ([*MSVC*](https://visualstudio.microsoft.com/de/vs/features/cplusplus/)) or
 * `__func__` (otherwise).
 * @details It can be used as compiler independent way to enable a better function name in the assertion macros.
 */
#ifdef __GNUG__
#include <execinfo.h>
#include <cxxabi.h>
#define SYCL_LSH_PRETTY_FUNC_NAME__ __PRETTY_FUNCTION__
#elif _MSVC_VER
#define SYCL_LSH_PRETTY_FUNC_NAME__ __FUNCSIG__
#else
#define SYCL_LSH_PRETTY_FUNC_NAME__ __func__
#endif

/**
 * @def SYCL_LSH_DEBUG_ASSERT
 * @brief Defines a custom `assert()` macro with potential additional parameters to the assertion message.
 * @details This macro is only defined if `SYCL_LSH_ENABLE_DEBUG` is set to `On` during [CMake's](https://cmake.org/) configuration step.
 * @param[in] cond the assert condition
 * @param[in] msg the custom assert message
 * @param[in] ... varying number of parameters to fill the `printf` like placeholders in the custom assert message
 *
 * @def SYCL_LSH_DEBUG0_ASSERT
 * @brief Defines a custom `assert()` macro.
 * @details This macro is only defined if `SYCL_LSH_ENABLE_DEBUG` is set to `On` during [CMake's](https://cmake.org/) configuration step.
 * @param[in] cond the assert condition
 * @param[in] msg the custom assert message
 */
#if SYCL_LSH_DEBUG
#define SYCL_LSH_DEBUG_ASSERT(cond, msg, ...)                                                                       \
  if (false) sycl_lsh::detail::check_args(msg, __VA_ARGS__);                                                        \
  sycl_lsh::detail::assert_function(__FILE__, SYCL_LSH_PRETTY_FUNC_NAME__, __LINE__, cond, #cond, msg, __VA_ARGS__)
#define SYCL_LSH_DEBUG0_ASSERT(cond, msg) sycl_lsh::detail::assert_function(__FILE__, SYCL_LSH_PRETTY_FUNC_NAME__, __LINE__, cond, #cond, msg)
#else
#define SYCL_LSH_DEBUG_ASSERT(cond, msg, ...)
#define SYCL_LSH_DEBUG0_ASSERT(cond, msg)
#endif

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP
