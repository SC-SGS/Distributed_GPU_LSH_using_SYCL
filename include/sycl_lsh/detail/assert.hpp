/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Defines a custom assertion macro with more intuitive syntax and better error message.
 */

#ifndef SYCL_LSH_DETAIL_ASSERT_HPP
#define SYCL_LSH_DETAIL_ASSERT_HPP
#pragma once

#include "sycl_lsh/exceptions/source_location.hpp"  // sycl_lsh::source_location
#include "sycl_lsh/mpi/detail/utility.hpp"          // SYCL_LSH_MPI_ERROR_CHECK
#include "sycl_lsh/mpi/environment.hpp"             // sycl_lsh::mpi::is_active

#include "fmt/color.h"   // fmt::emphasis, fmt::fg, fmt::color
#include "fmt/format.h"  // fmt::format
#include "mpi.h"     // MPI_Abort, MPI_COMM_WORLD

#include <cstdlib>      // std::abort
#include <iostream>     // std::cerr, std::endl
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::forward

namespace sycl_lsh::detail {

/**
 * @brief Function called by the `SYCL_LSH_ASSERT` macro. Checks the assertion condition. If the condition evaluates to `false`,
 *        prints the assertion condition together with additional information (e.g., `sycl_lsh::source_location` information) and aborts the program.
 * @tparam Args the placeholder types
 * @param[in] cond the assertion condition, aborts the program if evaluated to `false`
 * @param[in] cond_str the assertion condition as string
 * @param[in] loc the source location where the assertion appeared
 * @param[in] msg the custom assertion message
 * @param[in] args the placeholder values for the custom assertion message
 */
template <typename... Args>
void check_assertion(const bool cond, const std::string_view cond_str, const source_location &loc, const std::string_view msg, Args &&...args) {
    // check if the assertion holds
    if (!cond) {
        // print assertion error message
        std::cerr << fmt::format(
            "Assertion '{}' failed!\n"
            "{}"
            "  in file            {}\n"
            "  in function        {}\n"
            "  @ line             {}\n\n"
            "{}\n",
            fmt::format(fmt::emphasis::bold | fmt::fg(fmt::color::green), "{}", cond_str),
            loc.world_rank().has_value() ? fmt::format("  on MPI world rank  {}\n", loc.world_rank().value()) : std::string{},
            loc.file_name(),
            loc.function_name(),
            loc.line(),
            fmt::format(fmt::emphasis::bold | fmt::fg(fmt::color::red), msg, std::forward<Args>(args)...))
                  << std::endl;

        // abort further execution -> call MPI_Abort if in an MPI environment
        if (mpi::is_active()) {
            SYCL_LSH_MPI_ERROR_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
        } else {
            std::abort();
        }
    }
}

}  // namespace sycl_lsh::detail

/**
 * @def SYCL_LSH_ASSERT
 * @brief Defines a custom `assert()` macro with potential additional parameters to the assertion message.
 * @details This macro is only defined if `SYCL_LSH_ENABLE_DEBUG` is set to `On` during [CMake's](https://cmake.org/) configuration step.
 */
#if defined(SYCL_LSH_ASSERTS_ENABLED)
    #define SYCL_LSH_ASSERT(cond, msg, ...) sycl_lsh::detail::check_assertion((cond), (#cond), sycl_lsh::source_location::current(), (msg), ##__VA_ARGS__)
#else
    #define SYCL_LSH_ASSERT(cond, msg, ...)
#endif

#endif  // SYCL_LSH_DETAIL_ASSERT_HPP
