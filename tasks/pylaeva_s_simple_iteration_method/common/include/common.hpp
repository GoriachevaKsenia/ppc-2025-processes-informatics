#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace pylaeva_s_simple_iteration_method {

using InType = std::tuple<size_t, std::vector<double>, std::vector<double>>;  // matrix_size, a, b
using OutType = std::vector<double>;                                          // result vector
using TestType = std::string;                                                 // filename with test data
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace pylaeva_s_simple_iteration_method
