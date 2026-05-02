/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/exceptions/exceptions.hpp"

#include "sycl_lsh/exceptions/source_location.hpp"  // sycl_lsh::source_location

#include "fmt/format.h"  // fmt::format

#include <stdexcept>    // std::runtime_error
#include <string>       // std::string
#include <string_view>  // std::string_view

namespace sycl_lsh {

exception::exception(const std::string &msg, const std::string_view class_name, source_location loc) :
    std::runtime_error{ msg },
    class_name_{ class_name },
    loc_{ loc } { }

const source_location &exception::loc() const noexcept { return loc_; }

std::string exception::what_with_loc() const {
    return fmt::format(
        "{}\n"
        "{} thrown:\n"
        "  in file      {}\n"
        "  in function  {}\n"
        "  @ line       {}",
        this->what(),
        class_name_,
        loc_.file_name(),
        loc_.function_name(),
        loc_.line());
}

cmd_parser_exit::cmd_parser_exit(const int exit_code, source_location loc) :
    exception{ fmt::format("exit code: {}", exit_code), "cmd_parser_exit", loc },
    exit_code_{ exit_code } { }

not_implemented_exception::not_implemented_exception(const std::string &msg, source_location loc) :
    exception{ msg, "not_implemented_exception", loc } { }

mpi_exception::mpi_exception(const std::string &msg, source_location loc) :
    exception{ msg, "mpi_exception", loc } { }

file_exception::file_exception(const std::string &msg, source_location loc) :
    exception{ msg, "file_exception", loc } { }

file_parsing_exception::file_parsing_exception(const std::string &msg, source_location loc) :
    exception{ msg, "file_parsing_exception", loc } { }

}  // namespace sycl_lsh