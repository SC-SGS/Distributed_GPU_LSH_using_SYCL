/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-29
 *
 * @brief Defines a custom assertion macro with more intuitive syntax and better error message.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP

#include <stdio.h>

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
 */
#if SYCL_LSH_DEBUG
#define SYCL_LSH_DEBUG_ASSERT(cond, msg)                                                                        \
if (!(cond)) printf("Assertion '%s' failed!\n  in file     '%s'\n  in function '%s'\n  @ line     %i\n\n%s\n",  \
        #cond, __FILE__, SYCL_LSH_PRETTY_FUNC_NAME__, __LINE__, msg);
#else
#define SYCL_LSH_DEBUG_ASSERT(cond, msg)
#endif

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP
