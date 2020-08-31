/**
 * @file
 * @author Marcel Breyer
 * @date 2020-08-31
 *
 * @brief Defines a custom assertion macro with more intuitive syntax and better error message.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP

#include <cstdlib>
#include <utility>

#include <detail/print.hpp>
#include <detail/source_location.hpp>


namespace detail {

    /**
     * @brief
     * @tparam Args parameter pack for the placeholder types
     * @param[in] cond the assert condition, terminates the program if evaluated to `false`
     * @param[in] cond_str the assert condition as string for a better error message
     * @param[in] loc the location where the assertion has been triggered
     * @param[in] msg the custom assert message printed after the assertion location
     * @param[in] args the arguments to fill the `printf` like placeholders in the custom error message
     */
    template <typename... Args>
    inline void check(const bool cond, const char* cond_str, const source_location& loc, const char* msg, Args&&... args) {
        // check if the condition holds
        if (!cond) {
            // calculate sizes of the messages
            const char* loc_msg = loc.rank() == -1 ?
                                  "Assertion '{}' failed!\n  in file '{}'\n  in function '{}'\n  @ line {}\n\n" :
                                  "Assertion '{}' failed on rank {}!\n  in file '{}'\n  in function '{}'\n  @ line {}\n\n";
            const int loc_msg_size = detail::c_str_size(loc_msg);
            const int msg_size = detail::c_str_size(msg);

            // create new full_msg = source location msg + provided message + 2 newlines + null-terminator
            char* full_msg = new char[loc_msg_size + msg_size + 2 + 1];

            // copy source_location message to full_msg
            int idx = 0;
            for (; idx < loc_msg_size; ++idx) {
                full_msg[idx] = loc_msg[idx];
            }
            // copy provided message to full_msg
            for (int i = 0; i < msg_size; ++i, ++idx) {
                full_msg[idx] = msg[i];
            }
            // add trailing newlines and null-terminator
            full_msg[idx++] = '\n';
            full_msg[idx++] = '\n';
            full_msg[idx] = '\0';

            // print the full_msg
            if (loc.rank() == -1) {
                print(full_msg, cond_str, loc.file_name(), loc.function_name(), loc.line(), std::forward<Args>(args)...);
            } else {
                print(full_msg, cond_str, loc.rank(), loc.file_name(), loc.function_name(), loc.line(), std::forward<Args>(args)...);
            }

            // delete full_msg (previously allocated with new)
            delete[] full_msg;

// call abort if running on the CPU
#if SYCL_TARGET == 0
            abort();
#endif
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
 *
 * @def DEBUG_ASSERT_MPI
 * @brief Defines a custom `assert()` macro including the MPI rank info.
 * @details This macro is only defined in debug builds.
 * @param[in] comm_rank the MPI rank
 * @param[in] cond the assert condition
 * @param[in] msg the custom assert message
 * @param[in] ... varying number of parameters to fill the `printf` like placeholders in the custom assert message
 */
#ifdef NDEBUG
#define DEBUG_ASSERT(cond, msg, ...)
#define DEBUG_ASSERT_MPI(cond, comm_rank, msg, ...)
#else
#define DEBUG_ASSERT(cond, msg, ...) \
        detail::check(cond, #cond, detail::source_location::current(PRETTY_FUNC_NAME__, __FILE__, __LINE__), msg, __VA_ARGS__)
#define DEBUG_ASSERT_MPI(comm_rank, cond, msg, ...) \
        detail::check(cond, #cond, detail::source_location::current(comm_rank, PRETTY_FUNC_NAME__, __FILE__, __LINE__), msg, __VA_ARGS__)
#endif


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ASSERT_HPP
