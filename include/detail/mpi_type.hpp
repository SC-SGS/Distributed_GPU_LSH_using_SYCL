#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_TYPE_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_MPI_TYPE_HPP


#include <mpi.h>


/**
 * @def CREATE_MPI_TYPE_CAST
 * @brief Defines a macro to create all possible conversion functions from @p type to the *MPI_Datatype* @p mpi_type.
 * @param[in] type the data type to convert to a *MPI_Datatype*
 * @param[in] mpi_type the *MPI_Datatype*
 */
#define CREATE_MPI_TYPE_CAST(type, mpi_type)                                         \
template <>                                                                          \
[[nodiscard]] inline MPI_Datatype mpi_type_cast<type>() noexcept { return mpi_type; }


namespace detail {

    /**
     * @brief Tries to convert the given type to its corresponding *MPI_Datatype*.
     * @details The definition is marked as **deleted** if no conversion from `T` to a *MPI_Datatype* is possible.
     * @tparam T the type to convert to *MPI_Datatype*
     * @return the *MPI_Datatype* equivalent of `T`
     */
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
