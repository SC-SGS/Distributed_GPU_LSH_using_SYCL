/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements conversion functions from builtin types to their respective *MPI_Datatype* equivalents.
 */

#ifndef SYCL_LSH_MPI_DETAIL_TYPE_CAST_HPP
#define SYCL_LSH_MPI_DETAIL_TYPE_CAST_HPP
#pragma once

#include "mpi.h"  // MPI_Datatype, various MPI datatypes

#include <complex>      // std::complex
#include <type_traits>  // std::enable_if_t, std::is_enum_v, std::underlying_type_t

/**
 * @def SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING
 * @brief Defines a macro to create all possible conversion from a C++ type to a MPI_Datatype.
 * @param[in] cpp_type the C++ type
 * @param[in] mpi_type the corresponding MPI_Datatype
 */
#define SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(cpp_type, mpi_type) \
    template <>                                                  \
    [[nodiscard]] inline MPI_Datatype mpi_datatype<cpp_type>() { return mpi_type; }

namespace sycl_lsh::mpi::detail {

/**
 * @brief Tries to convert the given C++ type to its corresponding MPI_Datatype.
 * @details The definition is marked as **deleted** if `T` isn't representable as [`MPI_Datatype`](https://www.mpi-forum.org/docs/mpi-2.2/mpi22-report/node44.htm) or an enum.
 * @tparam T the type to convert to a MPI_Datatype
 * @return the corresponding MPI_Datatype (`[[nodiscard]]`)
 */
template <typename T, std::enable_if_t<!std::is_enum_v<T>, bool> = true>
[[nodiscard]] MPI_Datatype mpi_datatype() = delete;

SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(bool, MPI_C_BOOL)

// character types
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(char, MPI_CHAR)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(signed char, MPI_SIGNED_CHAR)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(unsigned char, MPI_UNSIGNED_CHAR)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(wchar_t, MPI_WCHAR)

// integer types
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(signed short, MPI_SHORT)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(unsigned short, MPI_UNSIGNED_SHORT)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(signed int, MPI_INT)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(unsigned int, MPI_UNSIGNED)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(signed long int, MPI_LONG)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(unsigned long int, MPI_UNSIGNED_LONG)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(signed long long int, MPI_LONG_LONG)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(unsigned long long int, MPI_UNSIGNED_LONG_LONG)
// SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(std::int8_t, MPI_INT8_T)
// SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(std::int16_t, MPI_INT16_T)
// SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(std::int32_t, MPI_INT32_T)
// SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(std::int64_t, MPI_INT64_T)
// SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(std::uint8_t, MPI_UINT8_T)
// SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(std::uint16_t, MPI_UINT16_T)
// SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(std::uint32_t, MPI_UINT32_T)
// SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(std::uint64_t, MPI_UINT64_T)

// floating point types
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(float, MPI_FLOAT)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(double, MPI_DOUBLE)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(long double, MPI_LONG_DOUBLE)

// complex types
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(std::complex<float>, MPI_C_COMPLEX)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(std::complex<double>, MPI_C_DOUBLE_COMPLEX)
SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING(std::complex<long double>, MPI_C_LONG_DOUBLE_COMPLEX)

/**
 * @brief Specialization for enums: for enums, use their underlying type in MPI communications.
 * @tparam T the enum type to convert to an MPI_Datatype
 * @return the corresponding MPI_Datatype (`[[nodiscard]]`)
 */
template <typename T, std::enable_if_t<std::is_enum_v<T>, bool> = true>
[[nodiscard]] MPI_Datatype mpi_datatype() {
    return mpi_datatype<std::underlying_type_t<T>>();
}

}  // namespace sycl_lsh::mpi::detail

#undef SYCL_LSH_CREATE_MPI_DATATYPE_MAPPING

#endif  // SYCL_LSH_MPI_DETAIL_TYPE_CAST_HPP
