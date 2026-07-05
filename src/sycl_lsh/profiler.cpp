/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 */

#include "sycl_lsh/profiler.hpp"

#include "sycl_lsh/constants.hpp"                    // sycl_lsh::real_type, sycl_lsh::index_type, sycl_lsh::hash_value_type
#include "sycl_lsh/data_set.hpp"                     // sycl_lsh::data_set::attributes
#include "sycl_lsh/detail/arithmetic_type_name.hpp"  // sycl_lsh::detail::arithmetic_type_name
#include "sycl_lsh/options.hpp"                      // sycl_lsh::options, sycl_lsh::locality_sensitive_hashing_options
#include "sycl_lsh/profiling_types.hpp"              // sycl_lsh::profiling_types

#include "fmt/chrono.h"  // fmt::gmtime
#include "fmt/format.h"  // fmt::format
#include "hws/core.hpp"  // hws::system_hardware_sampler

#include <cstddef>      // std::size_t
#include <ctime>        // std::time
#include <ostream>      // std::ostream, std::ofstream, std::endl
#include <string>       // std::string
#include <string_view>  // std::string_view

namespace {

/**
 * @brief Indent each newline with @p indentation.
 * @param[in] str the string to indent
 * @param[in] indentation the indentation
 * @return the new string with the correct indentation (`[[nodiscard]]`)
 */
[[nodiscard]] std::string indent_newlines(const std::string &str, const std::string_view indentation = "  ") {
    std::string result;
    std::size_t pos = 0;
    std::size_t prev = 0;
    result += indentation;
    while ((pos = str.find('\n', prev)) != std::string::npos) {
        result += str.substr(prev, pos - prev + 1);  // include the '\n'
        result += indentation;
        prev = pos + 1;
    }
    result += str.substr(prev);  // last line (no trailing '\n')
    return result;
}

}  // namespace

namespace sycl_lsh {

profiler::profiler(const profiling_types profiling_type) noexcept :
    profiling_type_{ profiling_type } {
    // initialize the hws hardware sampler only if requested
    if (profiling_type_ == profiling_types::hws) {
        using namespace std::chrono_literals;
        hardware_sampler_ = std::make_unique<hws::system_hardware_sampler>(SYCL_LSH_HARDWARE_SAMPLING_INTERVAL);
        hardware_sampler_->start_sampling();
    }
}

void profiler::add_entry(const std::string &group, const data_set::attributes &attr) {
    this->add_entry(group, "total_size", attr.total_size);
    this->add_entry(group, "rank_size", attr.rank_size);
    this->add_entry(group, "dims", attr.dims);
}

void profiler::add_entry(const locality_sensitive_hashing_options &opt) {
    this->add_entry("options", "hash_functions_type", opt.hash_function);
    this->add_entry("options", "hash_pool_size", opt.hash_pool_size);
    this->add_entry("options", "num_hash_functions", opt.num_hash_functions);
    this->add_entry("options", "num_hash_tables", opt.num_hash_tables);
    this->add_entry("options", "hash_table_size", opt.hash_table_size);
    this->add_entry("options", "w", opt.w);
    this->add_entry("options", "num_cut_off_points", opt.num_cut_off_points);
}

void profiler::add_entry(const options &opt) {
    this->add_entry("options", "n_neighbors", opt.n_neighbors);
    this->add_entry("options", "data_file", opt.data_file);
    this->add_entry("options", "file_parser", opt.file_parser);
    if (opt.indices_save_file.has_value()) {
        this->add_entry("options", "indices_save_file", opt.indices_save_file.value());
    }
    if (opt.distances_save_file.has_value()) {
        this->add_entry("options", "distances_save_file", opt.distances_save_file.value());
    }
    if (opt.indices_ground_truth_file.has_value()) {
        this->add_entry("options", "indices_ground_truth_file", opt.indices_ground_truth_file.value());
    }
    if (opt.distances_ground_truth_file.has_value()) {
        this->add_entry("options", "distances_ground_truth_file", opt.distances_ground_truth_file.value());
    }
    this->add_entry("options", "work_group_size", opt.work_group_size);
    this->add_entry(opt.lsh_options);
}

void profiler::add_event(const std::string &name) const {
    if (hardware_sampler_ != nullptr) {
        hardware_sampler_->add_event(name);
    }
}

void profiler::dump(const std::string &filename) {
    // if the profiling type is none, nothing to be done
    if (profiling_type_ != profiling_types::none) {
        std::ofstream out{ filename, std::ios::app };
        this->dump(out);
    }
}

void profiler::dump(std::ostream &out) {
    // if the profiling type is none, nothing to be done
    if (profiling_type_ != profiling_types::none) {
        // stop the hardware sampling if possible
        if (hardware_sampler_ != nullptr) {
            hardware_sampler_->stop_sampling();
        }

        // copy the internal entries such that adding the metadata does not change them
        auto entries{ entries_ };

        // add the metadata to the map copy
        // general metadata
        this->add_entry_impl(entries, "metadata", "build_type", std::string_view{ SYCL_LSH_BUILD_TYPE });
        this->add_entry_impl(entries, "metadata", "date", fmt::format("{:%Y-%m-%d %H:%M:%S}", fmt::gmtime(std::time(nullptr))));
        this->add_entry_impl(entries, "metadata", "timer", std::string_view{ SYCL_LSH_TIMER_NAME });
#if defined(SYCL_LSH_ASSERTS_ENABLED)
        this->add_entry_impl(entries, "metadata", "asserts", true);
#else
        this->add_entry_impl(entries, "metadata", "asserts", false);
#endif
#if defined(SYCL_LSH_RANDOM_NUMBERS_DEBUG)
        this->add_entry_impl(entries, "metadata", "random_number_debug", true);
#else
        this->add_entry_impl(entries, "meta_data", "random_number_debug", false);
#endif

        this->add_entry_impl(entries, "metadata", "real_type", detail::arithmetic_type_name<real_type>());
        this->add_entry_impl(entries, "metadata", "index_type", detail::arithmetic_type_name<index_type>());
        this->add_entry_impl(entries, "metadata", "hash_value_type", detail::arithmetic_type_name<hash_value_type>());
        this->add_entry_impl(entries, "metadata", "BLOCKING_SIZE", BLOCKING_SIZE);

        // backend related meta-data
        this->add_entry_impl(entries, "backend", "sycl_implementation", std::string_view{ SYCL_LSH_IMPLEMENTATION });
        this->add_entry_impl(entries, "backend", "target_arch", std::string_view{ SYCL_LSH_TARGET_ARCH });
#if defined(SYCL_LSH_CPU_VECTORIZATION_TARGET)
        this->add_entry_impl(entries, "backend", "cpu_vectorization_width", std::string_view{ SYCL_LSH_CPU_VECTORIZATION_TARGET });
#endif

        // output the data in a YAML format
        out << "---\n";
        for (const auto &[group, group_entries] : entries) {
            out << group << ":\n";
            for (const auto &[name, value] : group_entries) {
                out << "  " << name << ": " << value << '\n';
            }
            out << '\n';
        }
        // if available, add the hardware samples at the end
        if (hardware_sampler_ != nullptr) {
            for (const auto &sampler : hardware_sampler_->samplers()) {
                out << sampler->device_identification() << ":\n";
                out << indent_newlines(hardware_sampler_->as_yaml_string());
                out << '\n';
            }
        }
        out << std::endl;
    }
}

void profiler::clear_entries() {
    entries_.clear();
}

profiling_types profiler::profiling_type() const noexcept {
    return profiling_type_;
}

}  // namespace sycl_lsh
