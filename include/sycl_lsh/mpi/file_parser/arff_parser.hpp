/**
 * @file
 * @author Marcel Breyer
 * @date 2020-09-29
 *
 * @brief File parser for parsing `.arff` data files.
 */

#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP

#include <sycl_lsh/exceptions/not_implemented.hpp>
#include <sycl_lsh/mpi/communicator.hpp>
#include <sycl_lsh/mpi/file_parser/base_parser.hpp>
#include <sycl_lsh/mpi/logger.hpp>

#include <stdexcept>
#include <string_view>

namespace sycl_lsh::mpi {

    /**
     * @brief File parser class for the **arff** data format.
     * @details Not yet implemented.
     * @tparam Options  type of the used @ref sycl_lsh::options class
     * @tparam T the type of the data to parse
     */
    template <typename Options, typename T>
    class arff_parser final : public file_parser<Options, T> {
        using base_type = file_parser<Options, T>;
    public:
        /// The index type as specified in the provided @ref sycl_lsh::options template type.
        using index_type = typename base_type::index_type;
        /// The type of the data which should get parsed.
        using parsing_type = typename base_type::parsing_type;


        // ---------------------------------------------------------------------------------------------------------- //
        //                                                constructor                                                 //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Construct a new @ref sycl_lsh::mpi::arff_parser object responsible for parsing
         *        [`.arff´](https://www.cs.waikato.ac.nz/~ml/weka/arff.html) files.
         * @param[in] file_name the file to parse
         * @param[in] comm the used @ref sycl_lsh::mpi::communicator
         * @param[in] logger the used @ref sycl_lsh::mpi::logger
         *
         * @throws std::logic_error since `.arff` files aren't currently supported.
         */
        arff_parser(std::string_view file_name, const communicator& comm, const logger& logger);

        
        // ---------------------------------------------------------------------------------------------------------- //
        //                                                  parsing                                                   //
        // ---------------------------------------------------------------------------------------------------------- //
        /**
         * @brief Parse the **total** number of data points in the file.
         * @return the total number of data points (`[[nodiscard]]`)
         *
         * @throws sycl_lsh::not_implemented since `.arff` files aren't currently supported.
         */
        [[nodiscard]]
        index_type parse_total_size() const override;
        /**
         * @brief Parse the number of dimensions of each data point in the file.
         * @return the number of dimensions (`[[nodiscard]]`)
         *
         * @throws sycl_lsh::not_implemented since `.arff` files aren't currently supported.
         */
        [[nodiscard]]
        index_type parse_dims() const override;
        /**
         * @brief Parse the content of the file.
         * @param[out] buffer to write the data to
         *
         * @throws sycl_lsh::not_implemented since `.arff` files aren't currently supported.
         */
        void parse_content(parsing_type* buffer) const override;
        /**
         * @brief Write the content in @p buffer to the file.
         * @param buffer the data to write to the file
         *
         * @throws sycl_lsh::not_implemented since `.arff` files aren't currently supported.
         */
        void write_content(parsing_type* buffer) const override;

    };


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                constructor                                                 //
    // ---------------------------------------------------------------------------------------------------------- //
    template <typename Options, typename T>
    arff_parser<Options, T>::arff_parser(const std::string_view file_name, const communicator& comm, const logger& logger)
            : file_parser<Options, T>(file_name, comm, logger)
    {
        throw std::logic_error("Parsing an '.arff' file is currently no supported! Maybe run data_sets/convert_arff_to_binary.py first?");
    }


    // ---------------------------------------------------------------------------------------------------------- //
    //                                                  parsing                                                   //
    // ---------------------------------------------------------------------------------------------------------- //
    template <typename Options, typename T>
    [[nodiscard]]
    typename arff_parser<Options, T>::index_type arff_parser<Options, T>::parse_total_size() const {
        throw sycl_lsh::not_implemented();
    }

    template <typename Options, typename T>
    [[nodiscard]]
    typename arff_parser<Options, T>::index_type arff_parser<Options, T>::parse_dims() const {
        throw sycl_lsh::not_implemented();
    }

    template <typename Options, typename T>
    void arff_parser<Options, T>::parse_content([[maybe_unused]] parsing_type* buffer) const {
        throw sycl_lsh::not_implemented();
    }

    template <typename Options, typename T>
    void arff_parser<Options, T>::write_content([[maybe_unused]] parsing_type* buffer) const {
        throw sycl_lsh::not_implemented();
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_ARFF_PARSER_HPP
