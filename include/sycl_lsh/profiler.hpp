/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief A simple performance profile to track various metrics like options or runtimes.
 */

#ifndef SYCL_LSH_PROFILER_HPP
#define SYCL_LSH_PROFILER_HPP
#pragma once

#include "sycl_lsh/data_set.hpp"         // sycl_lsh::data_set::attributes
#include "sycl_lsh/options.hpp"          // sycl_lsh::options, sycl_lsh::locality_sensitive_hashing_options
#include "sycl_lsh/profiling_types.hpp"  // sycl_lsh::profiling_types

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/format.h"  // fmt::format
#include "fmt/std.h"     // format std:: types

#include <iosfwd>       // std::ostream forward declaration
#include <map>          // std::map
#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace sycl_lsh {

class profiler {
  public:
    /**
     * @brief Construct a new profiler using the @p profiling_type.
     * @details The profiling types are:
     *          - none: do not perform any profiling at all
     *          - runtimes: only profile runtimes (and some additional options like data_set sizes etc.)
     *          - hws: additionally to profiling runtimes, use the external hws library to also profile hardware characteristics
     * @param[in] profiling_type the used profiling capabilities
     */
    explicit profiler(profiling_types profiling_type) noexcept;

    /**
     * @brief Add all sycl_lsh::data_set::attributes @p attr to the @p group.
     * @param[in] group the group name
     * @param[in] attr the sycl_lsh::data_set::attributes
     */
    void add_entry(const std::string &group, const data_set::attributes &attr);
    /**
     * @brief Add all Locality Sensitive Hashing related options stored in @p opt as entries.
     * @param[in] opt the options to store
     */
    void add_entry(const locality_sensitive_hashing_options &opt);
    /**
     * @brief Add all options stored in @p opt as entries.
     * @param[in] opt the options to store
     */
    void add_entry(const options &opt);

    /**
     * @brief Add a new profiling entry to @p group with @p name and @p value.
     * @details The value is converted to a std::string using fmt.
     *          If @p T is either a std::string or std::string_view type, then the string value is enclosed in quotes.
     * @tparam T the type of the value to add
     * @param[in] group the group name
     * @param[in] name the value name
     * @param[in] value the value
     */
    template <typename T>
    void add_entry(const std::string &group, const std::string &name, const T &value) {
        this->add_entry_impl(entries_, group, name, value);
    }

    /**
     * @brief Dump all gathered profiling entries to the @p filename in a YAML format.
     * @param[in] filename the output YAML file name
     */
    void dump(const std::string &filename);
    /**
     * @brief Dump all gathered profiling entries to the @p out stream in a YAML format.
     * @param[in] out the output stream
     */
    void dump(std::ostream &out);

    /**
     * @brief Remove all already gathered entries.
     */
    void clear_entries();

  private:
    // The actual implementation. Passes the map as first parameter to be able to also add metadata to the map copy easily.
    template <typename T>
    void add_entry_impl(std::map<std::string, std::map<std::string, std::string>> &entries_map, const std::string &group, const std::string &name, const T &value) {
        // if the profiling type is none, nothing to be done
        if (profiling_type_ != profiling_types::none) {
            if (std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view>) {
                entries_map[group][name] = fmt::format("\"{}\"", value);
            } else {
                entries_map[group][name] = fmt::format("{}", value);
            }
        }
    }

    /// The currently used profiling type.
    profiling_types profiling_type_;

    /// All profiling entries currently gathered.
    std::map<std::string, std::map<std::string, std::string>> entries_;
};

}  // namespace sycl_lsh

#endif  // SYCL_LSH_PROFILER_HPP
