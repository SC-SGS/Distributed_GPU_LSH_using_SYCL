#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SOURCE_LOCATION_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SOURCE_LOCATION_HPP

#include <string>
#include <string_view>

#ifdef __GNUG__
#include <execinfo.h>
#include <cxxabi.h>
#define PRETTY_FUNC_NAME__ __PRETTY_FUNCTION__
#elif _MSC_VER
#define PRETTY_FUNC_NAME__ __FUNCSIG__
#else
#define PRETTY_FUNC_NAME__ __func__
#endif

namespace detail {

    class source_location {
    public:
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


        constexpr const std::string& file_name() const noexcept { return file_; }
        constexpr const std::string& function_name() const noexcept { return func_; }
        constexpr int line() const noexcept { return line_; }
        constexpr int column() const noexcept { return column_; }

    private:
        std::string file_ = "unknown";
        std::string func_ = "unknown";
        int line_ = 0;
        int column_ = 0;
    };

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_SOURCE_LOCATION_HPP
