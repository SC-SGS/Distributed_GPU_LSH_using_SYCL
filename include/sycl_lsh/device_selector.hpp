/**
 * @file
 * @author Marcel Breyer
 * @date 2020-today
 *
 * @brief Implements a device selector based on the `SYCL_LSH_TARGET` specified during [CMake](https://cmake.org/)'s configuration step.
 */

#ifndef SYCL_LSH_DEVICE_SELECTOR_HPP
#define SYCL_LSH_DEVICE_SELECTOR_HPP
#pragma once

#include "sycl/sycl.hpp"  // sycl::cpu_selector_v, sycl::cpu_selector_v

namespace sycl_lsh {

#if defined(SYCL_LSH_USE_CPU)
/// Select the default CPU device to run the SYCL kernels on.
constexpr auto device_selector = sycl::cpu_selector_v;
#else
/// Select the default GPU device to run the SYCL kernel on.
constexpr auto device_selector = sycl::gpu_selector_v;
#endif

}  // namespace sycl_lsh

#endif  // SYCL_LSH_DEVICE_SELECTOR_HPP
