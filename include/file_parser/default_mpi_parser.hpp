/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-16
 *
 * @brief File parser for parsing plain data files.
 */


#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_MPI_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_MPI_PARSER_HPP


#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>

#include <mpi.h>

#include <config.hpp>
#include <detail/convert.hpp>
#include <file_parser/base_file_parser.hpp>


/**
 * @brief Default file parser for parsing plain data files in a distributed fashion using MPI IO.
 * @details Expected file format:
 * @code
 * 5
 * 2
 * 0.2,0.4
 * 1.3,1.6
 * 2.4,2.8
 * 3.6,4.2
 * 3.8,4.6
 * @endcode
 * @tparam layout determines whether the data is saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 *
 * @note The file **must** be saved in **binary** form.
 */
template <memory_layout layout, typename Options>
class default_mpi_parser final : public file_parser<layout, Options> {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by a 'options' type!");

    /// The type of the base @ref file_parser.
    using base = file_parser<layout, Options>;
public:
    /// The type of the underlying data as specified as in the provided @ref options class.
    using real_type = typename Options::real_type;
    /// The index type as specified as in the provided @ref options class.
    using index_type = typename Options::index_type;

    /**
     * @brief Constructs a new @ref default_mpi_parser object for parsing plain data files.
     * @param[in] file the file to parse
     *
     * @throw std::invalid_argument if @p file doesn't exist
     */
    explicit default_mpi_parser(std::string file, const MPI_Comm& communicator)
        : file_parser<layout, Options>(std::move(file), true), comm_(communicator)
    {
        std::cout << "Parsing a file using the default_mpi_parser together with MPI IO!" << std::endl;

        MPI_File_open(comm_, base::file_.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file_);
    }
    /**
     * @brief Close the *MPI_File* handle upon destruction.
     */
    ~default_mpi_parser() {
        MPI_File_close(&file_);
    }

    [[nodiscard]] index_type parse_size() const override {
        // read first line containing the size
        int size;
        MPI_File_read_at(file_, 0, &size, 1, detail::mpi_type_cast<index_type>(), MPI_STATUS_IGNORE);
        return static_cast<index_type>(size);
    }
    [[nodiscard]] index_type parse_dims() const override {
        // read second line containing the dims
        int dims;
        MPI_File_read_at(file_, sizeof(index_type), &dims, 1, detail::mpi_type_cast<index_type>(), MPI_STATUS_IGNORE);
        return static_cast<index_type>(dims);
    }
    void parse_content(sycl::buffer<real_type, 1>& buffer, const index_type size, const index_type dims) const override {
        DEBUG_ASSERT(0 < size, "Illegal size!: {}", size);
        DEBUG_ASSERT(0 < dims, "Illegal number of dimensions!: {}", dims);

        // calculate communicator size and rank
        int comm_size, comm_rank;
        MPI_Comm_size(comm_, &comm_size);
        MPI_Comm_rank(comm_, &comm_rank);

        // check for correct real_type
        MPI_Offset file_size;
        MPI_File_get_size(file_, &file_size);
        file_size -= 2 * sizeof(index_type);
        assert((file_size == size * dims * sizeof(real_type)));

        // calculate byte offsets
        MPI_Offset initial_offset = 2 * sizeof(index_type);
        MPI_Offset rank_size = size / comm_size;
        if (static_cast<MPI_Offset>(comm_rank) < size % comm_size) ++rank_size;
        MPI_Offset rank_offset = (size / comm_size * comm_rank + std::min<MPI_Offset>(comm_rank, size % comm_size)) * dims * sizeof(real_type);

        // read data elements // TODO 2020-06-16 15:09 marcel: without copy?
        std::vector<real_type> tmp_buffer(rank_size * dims);
        MPI_File_read_at(file_, initial_offset + rank_offset, tmp_buffer.data(), tmp_buffer.size(), detail::mpi_type_cast<real_type>(), MPI_STATUS_IGNORE);

        // copy to buffer
        auto acc = buffer.template get_access<sycl::access::mode::discard_write>();
        for (index_type point = 0; point < size; ++point) {
            for (index_type dim = 0; dim < dims; ++dim) {
                acc[base::get_linear_id(point, dim, size, dims)] = tmp_buffer[point * dims + dim];
            }
        }
    }

private:
    const MPI_Comm& comm_;
    MPI_File file_;
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_DEFAULT_MPI_PARSER_HPP
