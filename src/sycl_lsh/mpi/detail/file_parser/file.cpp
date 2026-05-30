/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/mpi/detail/file_parser/file.hpp"

#include "sycl_lsh/detail/utility.hpp"         // sycl_lsh::detail::to_lower_case
#include "sycl_lsh/exceptions/exceptions.hpp"  // sycl_lsh::file_exception
#include "sycl_lsh/mpi/detail/utility.hpp"     // SYCL_LSH_MPI_ERROR_CHECK

#include <filesystem>   // std::filesystem::exists
#include <iomanip>      // std::ios::failbit
#include <iostream>     // std::ostream, std::istream
#include <memory>       // std::addressof
#include <string>       // std::string
#include <type_traits>  // std::underlying_type_t
#include <utility>      // std::exchange

namespace sycl_lsh::mpi::detail {

// ---------------------------------------------------------------------------------------------------------- //
//                                        constructors and destructor                                         //
// ---------------------------------------------------------------------------------------------------------- //
file::file(const std::string &file_name, const communicator &comm, const mode open_mode) {
    // if we are in read mode, the file must already exist
    if (open_mode == mode::read && !std::filesystem::exists(file_name)) {
        throw file_exception{ fmt::format("Can't open file '{}'!", file_name) };
    }

    // open the file
    const int err = MPI_File_open(comm, file_name.data(), static_cast<std::underlying_type_t<mode>>(open_mode), MPI_INFO_NULL, &file_);

    // if we are in write mode and the file already exists, delete and reopen it
    if (open_mode == mode::write && err != MPI_SUCCESS) {
        SYCL_LSH_MPI_ERROR_CHECK(MPI_File_delete(file_name.data(), MPI_INFO_NULL));
        SYCL_LSH_MPI_ERROR_CHECK(MPI_File_open(comm, file_name.data(), static_cast<std::underlying_type_t<mode>>(open_mode), MPI_INFO_NULL, &file_));
    }
}

file::file(file &&other) noexcept :
    file_{ std::exchange(other.file_, MPI_FILE_NULL) } { }

file::~file() {
    if (file_ != MPI_FILE_NULL) {
        MPI_File_close(&file_);
    }
}

// ---------------------------------------------------------------------------------------------------------- //
//                                            assignment operators                                            //
// ---------------------------------------------------------------------------------------------------------- //
file &file::operator=(file &&rhs) noexcept {
    if (this != std::addressof(rhs)) {
        if (file_ != MPI_FILE_NULL) {
            MPI_File_close(&file_);
        }
        file_ = std::exchange(rhs.file_, MPI_FILE_NULL);
    }
    return *this;
}

std::ostream &operator<<(std::ostream &out, const file::mode mode) {
    switch (mode) {
        case file::mode::read:
            return out << "read";
        case file::mode::write:
            return out << "write";
    }
    return out;
}

std::istream &operator>>(std::istream &in, file::mode &mode) {
    std::string str{};
    in >> str;
    // convert string to lower case representation
    str = sycl_lsh::detail::to_lower_case(str);

    if (str == "read") {
        mode = file::mode::read;
    } else if (str == "write") {
        mode = file::mode::write;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace sycl_lsh::mpi::detail
