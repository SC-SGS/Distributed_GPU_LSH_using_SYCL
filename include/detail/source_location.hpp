/**
 * @file
 * @author Marcel Breyer
 * @date 2020-08-27
 *
 * @brief Custom implementation for the [`std::source_location`](https://en.cppreference.com/w/cpp/utility/source_location) class.
 * @details Includes a better function name (if supported) and the MPI rank.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SOURCE_LOCATION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SOURCE_LOCATION_HPP


/**
 * @def PRETTY_FUNC_NAME__
 * @brief The @ref PRETTY_FUNC_NAME__ macro is defined as `__PRETTY_FUNC__` ([*GCC*](https://gcc.gnu.org/) and
 * [*clang*](https://clang.llvm.org/)), `__FUNCSIG__` ([*MSVC*](https://visualstudio.microsoft.com/de/vs/features/cplusplus/)) or
 * `__func__` (otherwise).
 * @details It can be used as compiler independent way to enable a better function name when used as first parameter to
 * @ref detail::source_location::current().
 */
#ifdef __GNUG__
#define PRETTY_FUNC_NAME__ __PRETTY_FUNCTION__
#elif _MSC_VER
#define PRETTY_FUNC_NAME__ __FUNCSIG__
#else
#define PRETTY_FUNC_NAME__ __func__
#endif

/**
 * @def BUILTIN_FUNCTION__
 * @brief The `BUILTIN_FUNCTION__` macro is defined as `__builtin_FUNCTION` if supported, otherwise it's defined as a lambda
 *        returning `"unknown"`.
 */
#if __has_builtin(__builtin_FUNCTION)
#define BUILTIN_FUNCTION__ __builtin_FUNCTION
#else
#define BUILTIN_FUNCTION__ []() { return "unknown"; }
#endif

/**
 * @def BUILTIN_FILE__
 * @brief The `BUILTIN_FILE__` macro is defined as `__builtin_FILE` if supported, otherwise it's defined as a lambda
 *        returning `"unknown"`.
 */
#if __has_builtin(__builtin_FILE)
#define BUILTIN_FILE__ __builtin_FILE
#else
#define BUILTIN_FILE__ []() { return "unknown"; }
#endif

/**
 * @def BUILTIN_LINE__
 * @brief The `BUILTIN_LINE__` macro is defined as `__builtin_LINE` if supported, otherwise it's defined as a lambda returning `0`.
 */
#if __has_builtin(__builtin_LINE)
#define BUILTIN_LINE__ __builtin_LINE
#else
#define BUILTIN_LINE__ []() { return 0; }
#endif


namespace detail {

    /**
     * @brief Represents information of a specific source code location.
     */
    class source_location {
    public:
        /**
         * @brief Constructs a new source_location with the respective information about the current call side.
         * @details Sets the MPI rank to `-1` as none was specified.
         * @param[in] func the function name (including its signature if supported via the macro `PRETTY_FUNC_NAME__`)
         * @param[in] file the file name (absolute path)
         * @param[in] line the line number
         * @param[in] column the column number
         * @return the source_location holding the call side location information
         *
         * @attention @p column is always (independent of the call side position) default initialized to 0!
         */
        static source_location current(
                const char* func = BUILTIN_FUNCTION__(),
                const char* file = BUILTIN_FILE__(),
                const int line = BUILTIN_LINE__(),
                const int column = 0
        ) noexcept {
            return current(-1, func, file, line, column);
        }
        /**
         * @brief Constructs a new source_location with the respective information about the current call side including the MPI rank.
         * @param[in] comm_rank the MPI rank
         * @param[in] func the function name (including its signature if supported via the macro `PRETTY_FUNC_NAME__`)
         * @param[in] file the file name (absolute path)
         * @param[in] line the line number
         * @param[in] column the column number
         * @return the source_location holding the call side location information
         *
         * @attention @p column is always (independent of the call side position) default initialized to 0!
         */
        static source_location current(
                const int comm_rank,
                const char* func = BUILTIN_FUNCTION__(),
                const char* file = BUILTIN_FILE__(),
                const int line = BUILTIN_LINE__(),
                const int column = 0
        ) noexcept {
            source_location loc;
            loc.comm_rank_ = comm_rank;
            loc.file_ = file;
            loc.func_ = func;
            loc.line_ = line;
            loc.column_ = column;
            return loc;
        }

        /**
         * @brief Returns the absolute path name of the file.
         * @return the file name (`[[nodiscard]]`)
         */
        [[nodiscard]] constexpr const char* file_name() const noexcept { return file_; }
        /**
         * @brief Returns the function name without additional signature information (i.e. return type or parameters).
         * @return the function name (`[[nodiscard]]`)
         */
        [[nodiscard]] constexpr const char* function_name() const noexcept { return func_; }
        /**
         * @brief Returns the line number.
         * @return the line number (`[[nodiscard]]`)
         */
        [[nodiscard]] constexpr int line() const noexcept { return line_; }
        /**
         * @brief Returns the column number.
         * @return the column number (`[[nodiscard]]`)
         *
         * @attention Default value in @ref detail::source_location::current() always 0!
         */
        [[nodiscard]] constexpr int column() const noexcept { return column_; }
        /**
         * @brief Returns the current MPI rank.
         * @details Returns `-1` if no MPI rank has been specified.
         * @return the MPI rank (`[[nodiscard]]`)
         */
        [[nodiscard]] constexpr int rank() const noexcept { return comm_rank_; }

    private:
        const char* file_ = "unknown";
        const char* func_ = "unknown";
        int line_ = 0;
        int column_ = 0;
        int comm_rank_ = -1;
    };

}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SOURCE_LOCATION_HPP
