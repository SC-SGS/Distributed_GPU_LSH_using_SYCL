/**
 * @file
 * @author Marcel Breyer
 * @date 2020-06-17
 *
 * @brief Base class for all file parsers.
 * @details Pure virtual.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_FILE_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_FILE_PARSER_HPP


#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <config.hpp>
#include <detail/assert.hpp>
#include <options.hpp>
#include <mpi_buffer.hpp>


/**
 * @brief Pure virtual base class for all file parser.
 * @tparam layout determines whether the data is saved as *Array of Structs* or *Struct of Arrays*
 * @tparam Options represents various constant options to alter the algorithm's behaviour
 */
template <memory_layout layout, typename Options>
class file_parser {
    static_assert(std::is_base_of_v<detail::options_base, Options>, "The second template parameter must by an 'options' type!");
public:
    /// The type of the underlying data as specified as in the provided @ref options class.
    using real_type = typename Options::real_type;
    /// The index type as specified as in the provided @ref options class.
    using index_type = typename Options::index_type;

    /**
     * @brief Constructs a new @ref file_parser object and opens the file to retrieve a *MPI_File* handle.
     * @param[in] file the file to parse
     * @param[in] communicator the *MPI_Comm* communicator used to open the @p file with
     *
     * @throw std::invalid_argument if @p file doesn't exist
     */
    file_parser(const std::string& file, const MPI_Comm& communicator, const int comm_rank) : comm_(communicator), comm_rank_(comm_rank) {
        // check if the file exists
        if (!std::filesystem::exists(file)) {
            throw std::invalid_argument("File '" + file + "' doesn't exist!");
        }
        // open the file
        MPI_File_open(comm_, file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file_);
    }
    /**
     * @brief Close the *MPI_File* handle upon destruction.
     */
    virtual ~file_parser() {
        MPI_File_close(&file_);
    }

    /**
     * @brief Pareses the **total** number of data points in the file.
     * @return the total number of data points (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual index_type parse_size() const = 0;
    /**
     * @brief Parses the number of data points **per MPI rank** of the file.
     * @details Example: \n
     *          `this->parse_size() = 14, comm_size = 4` \n
     *          -> `this->parse_rank_size() = 4` for **ALL** MPI ranks and **NOT** `{ 4, 4, 3, 3 }`.
     * @return the number of data points per MPI rank (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual index_type parse_rank_size() const = 0;
    /**
     * @brief Parses the number of dimensions of each data point in the file.
     * @return the number of dimensions (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual index_type parse_dims() const = 0;
    /**
     * @brief Parse the content of the file.
     * @return the @ref mpi_buffers containing the parsed data (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual mpi_buffers<real_type, index_type> parse_content() const = 0;

protected:
    /**
     * @brief Converts a two-dimensional index into a flat one-dimensional index based on the current @ref memory_layout.
     * @param[in] point the provided data point
     * @param[in] dim the provided dimension
     * @param[in] size the number of data points
     * @param[in] dims the number of dimensions per data points
     * @return the flattened index (`[[nodiscard]]`)
     *
     * @pre @p point **must** be greater or equal than `0` and less than @p size.
     * @pre @p dim **must** be greater or equal than `0` and less than @p dims.
     */
    [[nodiscard]] constexpr index_type get_linear_id(const index_type point, const index_type dim,
            [[maybe_unused]] const index_type size, [[maybe_unused]] const index_type dims) const noexcept
    {
        DEBUG_ASSERT_MPI(comm_rank_, 0 <= point && point < size, "Out-of-bounce access!: 0 <= {} < {}", point, size);
        DEBUG_ASSERT_MPI(comm_rank_, 0 <= dim && dim < dims, "Out-of-bounce access!: 0 <= {} < {}", dim, dims);

        if constexpr (layout == memory_layout::aos) {
            // Array of Structs
            return dim + point * dims;
        } else {
            // Struct of Arrays
            return point + dim * size;
        }
    }

    /// The communicator used to open the *MPI_File*.
    const MPI_Comm& comm_;
    /// The current MPI rank.
    const int comm_rank_;
    /// The opened *MPI_File* handle.
    MPI_File file_;
};


#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_BASE_FILE_PARSER_HPP
