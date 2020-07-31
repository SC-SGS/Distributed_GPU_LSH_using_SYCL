/**
 * @file
 * @author Marcel Breyer
 * @date 2020-07-31
 *
 * @brief Implements a @ref options class for managing hyperparameters.
 */

#ifndef DISTRIBUTED_GPU_LSH_USING_SYCL_OPTIONS_HPP
#define DISTRIBUTED_GPU_LSH_USING_SYCL_OPTIONS_HPP

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <boost/type_index.hpp>

#include <config.hpp>
#include <detail/convert.hpp>
#include <hash_function.hpp>


/**
 * @brief Class containing all hyperparameters to change the behaviour of the algorithm.
 * @tparam real_t a floating point type
 * @tparam index_t an integer type
 * @tparam hash_value_t an integer type (used as type for the hash value)
 */
template <typename real_t = float, typename index_t = std::uint32_t, typename hash_value_t = std::uint32_t>
struct options : detail::options_base {
    // check template parameter types
    static_assert(std::is_floating_point_v<real_t>, "The first template parameter must be a floating point type!");
    static_assert(std::is_integral_v<index_t>, "The second template parameter must be an integral type!");
    static_assert(std::is_integral_v<hash_value_t>, "The third template parameter must be an integral type!");
public:
    // ---------------------------------------------------------------------------------------------------------- //
    //                                                type options                                                //
    // ---------------------------------------------------------------------------------------------------------- //

    /// The used floating point type.
    using real_type = real_t;
    /// The used integer type.
    using index_type = index_t;
    /// The used type of a hash value.
    using hash_value_type = hash_value_t;

    /**
     * @brief Factory class to create a new @ref options instance.
     */
    class factory {
        /// Befriend options class.
        template <typename, typename, typename>
        friend struct options;
    public:
        /**
         * @brief Construct a factory object with default values.
         * @param[in] comm_rank the current MPI rank
         */
        explicit factory(const int comm_rank) : comm_rank_(comm_rank) { }
        /**
         * @brief Construct a factory object using the values given in @p file.
         * @param[in] file the file containing the option values
         * @param[in] comm_rank the current MPI rank
         *
         * @throw std::invalid_argument if @p file doesn't exist.
         * @throw std::invalid_arguemnt if @p file contains a wrong option pair.
         */
        explicit factory(int comm_rank, const std::string& file) : comm_rank_(comm_rank) {
            // check if file exists
            if (!std::filesystem::exists(file)) {
                throw std::invalid_argument("File '" + file + "' doesn't exist!");
            }

            std::ifstream in(file);
            std::string opt, value;

            // try to read all options given in file
            while(in >> opt >> value) {
                if (opt == "hash_pool_size") {
                    this->set_hash_pool_size(detail::convert_to<index_type>(value));
                } else if (opt == "num_cut_off_points") {
                    this->set_num_cut_off_points(detail::convert_to<index_type>(value));
                } else if (opt == "num_hash_tables") {
                    this->set_num_hash_tables(detail::convert_to<index_type>(value));
                } else if (opt == "hash_table_size") {
                    this->set_hash_table_size(detail::convert_to<hash_value_type>(value));
                } else if (opt == "num_hash_functions") {
                    this->set_num_hash_functions(detail::convert_to<index_type>(value));
                } else if (opt == "num_multi_probes") {
                    this->set_num_multi_probes(detail::convert_to<index_type>(value));
                } else if (opt == "w") {
                    this->set_w(detail::convert_to<real_type>(value));
                } else if (opt == "real_type" || opt == "index_type" || opt == "hash_value_type"
                        || opt == "hash_functions_type" || opt == "probing_type")
                {
                    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                } else {
                    std::stringstream ss;
                    ss << "Invalid option '" << opt << " " << "' in file '" << file << "'.";
                    throw std::invalid_argument(ss.str());
                }
            }
        }

        /**
         * @brief Set the new number of hash functions in the hash pool.
         * @param[in] factory_hash_pool_size the number of hash functions in the hash pool
         * @return `*this`
         *
         * @pre @p factory_hash_pool_size **must** be greater than `0`.
         *
         * @throws std::invalid_argument if @p factory_hash_pool_size isn't greater than `0`
         */
        factory& set_hash_pool_size(const index_type factory_hash_pool_size) {
            if (factory_hash_pool_size <= 0) {
                throw std::invalid_argument("The number of hash functions in the hash pool must be greater than 0!");
            }
            hash_pool_size_ = factory_hash_pool_size;
            return *this;
        }
        /**
         * @brief Set the new number of cut-of points.
         * @param[in] factory_num_cut_off_points the number of cut-off points used in the entropy-based hash functions
         * @return `*this`
         *
         * @pre @p factory_num_cut_off_points **must** be greater than `0`
         *
         * @throws std::invalid_argument if @p factory_num_cut_off_points isn't greater than `0`
         */
        factory& set_num_cut_off_points(const index_type factory_num_cut_off_points) {
            if (factory_num_cut_off_points <= 0) {
                throw std::invalid_argument("The number of cut-off points used by the entropy-based hash functions must be greater than 0!");
            }
            num_cut_off_points_ = factory_num_cut_off_points;
            return *this;
        }
        /**
         * @brief Set the new number of hash tables to create.
         * @param[in] factory_num_hash_tables number of hash tables
         * @return `*this`
         *
         * @pre @p factory_num_hash_tables **must** be greater than `0`.
         *
         * @throws std::invalid_argument if @p factory_num_hash_tables isn't greater than `0`
         */
        factory& set_num_hash_tables(const index_type factory_num_hash_tables) {
            if (factory_num_hash_tables <= 0) {
                throw std::invalid_argument("The number of hash tables must be greater than 0!");
            }
            num_hash_tables_ = factory_num_hash_tables;
            return *this;
        }
        /**
         * @brief Set the new size of each hash table.
         * @details @p factory_hash_table_size should be a prime number.
         * @param[in] factory_hash_table_size hash table size
         * @return `*this`
         *
         * @pre @p factory_hash_table_size **must** be greater than `0`.
         * @pre @p factory_hash_table_size **must** be a prime number.
         *
         * @throws std::invalid_argument if @p factory_hash_table_size isn't greater than `0` or isn't a prime number
         */
        factory& set_hash_table_size(const hash_value_type factory_hash_table_size) {
            if (factory_hash_table_size <= 0) {
                throw std::invalid_argument("The size of a hash table must be greater than 0!");
            }
            if (!this->is_prime(factory_hash_table_size)) {
                throw std::invalid_argument("The size of a hash table must be a prime number!");
            }
            hash_table_size_ = factory_hash_table_size;
            return *this;
        }
        /**
         * @brief Set the new number of hash functions per hash table.
         * @param[in] factory_num_hash_functions number of hash functions
         * @return `*this`
         *
         * @pre @p factory_num_hash_functions **must** be greater than `0`.
         *
         * @throws std::invalid_argument if @p factory_num_hash_functions isn't greater than `0`
         */
        factory& set_num_hash_functions(const index_type factory_num_hash_functions) {
            if (factory_num_hash_functions <= 0) {
                throw std::invalid_argument("The number of hash functions per hash table must be greater than 0!");
            }
            num_hash_functions_ = factory_num_hash_functions;
            return *this;
        }
        /**
         * @brief Set the new number of multi-probes to perform per hash table.
         * @param[in] factory_num_multi_probes number of multi-probes
         * @return `*this`
         *
         * @pre @p factory_num_multi_probes **must** be greater than `0`.
         *
         * @throws std::invalid_argument if @p factory_num_multi_probes isn't greater than `0`
         */
        factory& set_num_multi_probes(const index_type factory_num_multi_probes) {
            if (factory_num_multi_probes <= 0) {
                throw std::invalid_argument("The number of multi-probes per hash table must be greater than 0!");
            }
            num_multi_probes_ = factory_num_multi_probes;
            return *this;
        }
        /**
         * @brief Set the new w value used in the hash value calculation: \f$h_{a, b} = \frac{a \cdot x + b}{w}\f$.
         * @param[in] factory_w constant value for the hash value calculation
         * @return `*this`
         *
         * @pre @p factory_w **must** be greater than `0.0`.
         *
         * @throws std::invalid_argument if @p factory_w isn't greater than `0.0`
         */
        factory& set_w(const real_type factory_w) {
            if (factory_w <= 0.0) {
                throw std::invalid_argument("The value of w must be greater than 0.0!");
            }
            w_ = factory_w;
            return *this;
        }

        /**
         * @brief Create a new @ref options object.
         * @return an @ref options object (`[[nodiscard]]`)
         */
        [[nodiscard]] options<real_type, index_type, hash_value_type> create() const {
            return options(*this);
        }

    private:
#ifndef NDBEUG
        /**
         * @brief Checks whether @p n is a prime number.
         * @details See: [https://en.wikipedia.org/wiki/Primality_test](https://en.wikipedia.org/wiki/Primality_test).
         * @param[in] n the number to check
         * @return `true` if @p n is prime, `false` otherwise
         */
        bool is_prime(const index_type n) {
            if (n <= 3) return n > 1;
            else if (n % 2 == 0 || n % 3 == 0) return false;
            for (index_type i = 5; i * i <= n; i += 6) {
                if (n % i == 0 || n % (i + 2) == 0) return false;
            }
            return true;
        }
#endif
        // TODO 2020-04-30 15:31 marcel: set meaningful defaults
        index_type hash_pool_size_ = static_cast<index_type>(10);
        index_type num_cut_off_points_ = static_cast<index_type>(10);
        index_type num_hash_tables_ = static_cast<index_type>(2);
        hash_value_type hash_table_size_ = static_cast<hash_value_type>(105613);
        index_type num_hash_functions_ = static_cast<index_type>(4);
        index_type num_multi_probes_ = static_cast<index_type>(6);
        real_type w_ = static_cast<real_type>(1.0);

        int comm_rank_;
    };


    /**
     * @brief Create a new options instance from a options factory.
     * @param[in] fact a options factory
     */
    options(options::factory fact)
            : hash_pool_size(fact.hash_pool_size_), num_cut_off_points(fact.num_cut_off_points_),
              num_hash_tables(fact.num_hash_tables_), hash_table_size(fact.hash_table_size_),
              num_hash_functions(fact.num_hash_functions_), num_multi_probes(fact.num_multi_probes_), w(fact.w_)
    {
        // TODO 2020-07-30 16:48 marcel: meaningful restriction?
        if constexpr (std::is_same_v<decltype(probing_type), decltype(probing::multiple)>) {
            if (num_multi_probes > num_hash_functions) {
                throw std::invalid_argument("The number of multi-probes can't be greater than the number of hash functions!");
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------------- //
    //                                              runtime options                                               //
    // ---------------------------------------------------------------------------------------------------------- //

    /// The number of hash functions int the hash pool.
    const index_type hash_pool_size;
    /// The number of cut-off points used in the entropy-based hash functions.
    const index_type num_cut_off_points;
    /// The number of hash tables to create.
    const index_type num_hash_tables;
    /// The size of each hash table (should be a prime).
    const hash_value_type hash_table_size;
    /// The number of hash functions per hash table.
    const index_type num_hash_functions;
    /// The number of multi-probes to perform per hash table.
    const index_type num_multi_probes;
    /// A constant used in the hash functions: \f$h_{a, b} = \frac{a \cdot x + b}{w}\f$.
    const real_type w;

    // ---------------------------------------------------------------------------------------------------------- //
    //                                            compile time options                                            //
    // ---------------------------------------------------------------------------------------------------------- //

    /// The type of the hash functions.
    static constexpr auto hash_functions_type = hash_functions::random_projection;
    /// Specifies whether multi-probe LSH should be used.
    static constexpr auto probing_type = probing::single;


    /**
     * @brief Saves the current options to @p file.
     * @details The content of @p file is overwritten if it already exists.
     * @param[in] file the name of the options @p file
     *
     * @throw std::invalid_argument if @p file can't be opened or created.
     */
    void save(const std::string& file) const {
        std::ofstream out(file, std::ofstream::trunc);
        if (out.bad()) {
            // something went wrong while opening/creating the file
            throw std::invalid_argument("Can't write to file '" + file + "'!");
        }
        out << *this << std::endl;
    }

    /**
     * @brief Print all options set in @p opt to the output stream @p out.
     * @param[inout] out the output stream to print @p opt
     * @param[in] opt the options
     * @return the output stream
     */
    friend std::ostream& operator<<(std::ostream& out, const options& opt) {
        out << "real_type '" << boost::typeindex::type_id<real_type>().pretty_name() << "'\n";
        out << "index_type '" << boost::typeindex::type_id<index_type>().pretty_name() << "'\n";
        out << "hash_value_type '" << boost::typeindex::type_id<hash_value_type>().pretty_name() << "'\n";
        out << "hash_functions_type '" << options::hash_functions_type << "'\n";
        out << "probing_type '" << options::probing_type << "'\n";
        out << "hash_pool_size " << opt.hash_pool_size << '\n';
        out << "num_cut_off_points " << opt.num_cut_off_points << '\n';
        out << "num_hash_tables " << opt.num_hash_tables << '\n';
        out << "hash_table_size " << opt.hash_table_size << '\n';
        out << "num_hash_functions " << opt.num_hash_functions << '\n';
        out << "num_multi_probes " << opt.num_multi_probes << '\n';
        out << "w " << opt.w;

        return out;
    }
};

#endif //DISTRIBUTED_GPU_LSH_USING_SYCL_OPTIONS_HPP
