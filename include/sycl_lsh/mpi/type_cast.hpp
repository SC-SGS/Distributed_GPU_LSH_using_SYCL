/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-21
 *
 * @brief Implements conversion functions from builtin types to their respective *MPI_Datatype* equivalents.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_TYPE_CAST_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_TYPE_CAST_HPP

#include <mpi.h>

/**
 * @def SYCL_LSH_MPI_CREATE_TYPE_CAST
 * @brief Defines a macro to create all possible conversion functions from @p type to the *MPI_Datatype* @p mpi_type.
 * @param[in] type the data type to convert to a *MPI_Datatype*
 * @param[in] mpi_type the *MPI_Datatype*
 */
#define SYCL_LSH_MPI_CREATE_TYPE_CAST(type, mpi_type)               \
template <>                                                         \
[[nodiscard]]                                                       \
inline MPI_Datatype type_cast<type>() noexcept { return mpi_type; }

namespace sycl_lsh::mpi {

    /**
     * @brief Tries to convert the given type to its corresponding *MPI_Datatype*.
     * @details The definition is marked as **deleted** if no conversion from `T` to a *MPI_Datatype* is possible.
     * @tparam T the type to convert to *MPI_Datatype*
     * @return the *MPI_Datatype* equivalent of `T`
     */
    template <typename T>
    [[nodiscard]]
    inline MPI_Datatype type_cast() noexcept = delete;

    // integer types
    SYCL_LSH_MPI_CREATE_TYPE_CAST(int, MPI_INT)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(long, MPI_LONG)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(short, MPI_SHORT)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(unsigned short, MPI_UNSIGNED_SHORT)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(unsigned, MPI_UNSIGNED)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(unsigned long, MPI_UNSIGNED_LONG)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(long long int, MPI_LONG_LONG_INT)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(unsigned long long, MPI_UNSIGNED_LONG_LONG)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(char, MPI_CHAR)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(signed char, MPI_SIGNED_CHAR)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(unsigned char, MPI_UNSIGNED_CHAR)

    // floating point types
    SYCL_LSH_MPI_CREATE_TYPE_CAST(float, MPI_FLOAT)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(double, MPI_DOUBLE)
    SYCL_LSH_MPI_CREATE_TYPE_CAST(long double, MPI_LONG_DOUBLE)

}

#undef SYCL_LSH_MPI_CREATE_TYPE_CAST

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_TYPE_CAST_HPP
