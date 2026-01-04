#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace romanova_v_jacobi_method {

using InType = std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<double>, double,
                          size_t>;  // x, A, b, eps, iterations
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace romanova_v_jacobi_method
