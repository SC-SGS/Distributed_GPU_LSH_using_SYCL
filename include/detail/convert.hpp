#ifndef DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERT_HPP
#define DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERT_HPP

#include <charconv>
#include <string>
#include <type_traits>

namespace detail {

    template <typename T>
    T convert_to(const std::string& str) {
        if constexpr (std::is_floating_point_v<std::decay_t<T>>) {
            try {
                if constexpr (std::is_same_v<std::decay_t<T>, float>) {
                    return std::stof(str);
                } else if constexpr (std::is_same_v<std::decay_t<T>, double>) {
                    return std::stod(str);
                } else {
                    return std::stold(str);
                }
            } catch (const std::exception& e) {
                throw std::invalid_argument("Can't parse string!: " + str);
            }
        } else {
            T val;
            auto [p, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
            if (ec == std::errc()) {
                return val;
            } else {
                throw std::invalid_argument("Can't parse string!: " + str);
            }
        }
    }

}

#endif // DISTRIBUTED_GPU_LSH_IMPLEMENTATION_USING_SYCL_CONVERT_HPP
