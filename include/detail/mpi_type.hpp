#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_TYPE_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_TYPE_HPP

#include <cstdint>

#include <mpi.h>


#define CREATE_MPI_TYPE_CAST(type, mpi_type)                                  \
template <>                                                                   \
[[nodiscard]] inline MPI_Datatype mpi_type_cast<type>() noexcept { return mpi_type; }


namespace detail {

    template <typename T>
    [[nodiscard]] inline MPI_Datatype mpi_type_cast() noexcept = delete;

    // integer types
    CREATE_MPI_TYPE_CAST(int, MPI_INT)
    CREATE_MPI_TYPE_CAST(long, MPI_LONG)
    CREATE_MPI_TYPE_CAST(short, MPI_SHORT)
    CREATE_MPI_TYPE_CAST(unsigned short, MPI_UNSIGNED_SHORT)
    CREATE_MPI_TYPE_CAST(unsigned, MPI_UNSIGNED)
    CREATE_MPI_TYPE_CAST(unsigned long, MPI_UNSIGNED_LONG)
    CREATE_MPI_TYPE_CAST(long long int, MPI_LONG_LONG_INT)
    CREATE_MPI_TYPE_CAST(unsigned long long, MPI_UNSIGNED_LONG_LONG)
    CREATE_MPI_TYPE_CAST(signed char, MPI_SIGNED_CHAR)
    CREATE_MPI_TYPE_CAST(unsigned char, MPI_UNSIGNED_CHAR)

    // floating point types
    CREATE_MPI_TYPE_CAST(float, MPI_FLOAT)
    CREATE_MPI_TYPE_CAST(double, MPI_DOUBLE)
    CREATE_MPI_TYPE_CAST(long double, MPI_LONG_DOUBLE)

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_TYPE_HPP
