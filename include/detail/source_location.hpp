/**
 * @file
 * @author Marcel Breyer
 * @date 2020-05-05
 *
 * @brief
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SOURCE_LOCATION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SOURCE_LOCATION_HPP

#include <string>
#include <string_view>

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


namespace detail {

    /**
     * @brief Represents information of a specific source code location.
     */
    class source_location {
    public:
        /**
         * @brief Constructs a new source_location with the respective information about the current call side.
         * @param[in] func the function name (including its signature if supported via the macro `PRETTY_FUNC_NAME__`)
         * @param[in] file the file name (absolute path)
         * @param[in] line the line number
         * @param[in] column the column number
         * @return the source_location holding the call side location information
         *
         * @attention @p column is always (independent of the call side position) default initialized to 0!
         */
        static source_location current(
                const std::string_view func = __builtin_FUNCTION(),
                const std::string_view file = __builtin_FILE(),
                const int line = __builtin_LINE(),
                const int column = 0
        ) noexcept {
            source_location loc;
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
        [[nodiscard]] constexpr const std::string& file_name() const noexcept { return file_; }
        /**
         * @brief Returns the function name without additional signature information (i.e. return type or parameters).
         * @return the function name (`[[nodiscard]]`)
         */
        [[nodiscard]] constexpr const std::string& function_name() const noexcept { return func_; }
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

    private:
        std::string file_ = "unknown";
        std::string func_ = "unknown";
        int line_ = 0;
        int column_ = 0;
    };

}


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SOURCE_LOCATION_HPP
