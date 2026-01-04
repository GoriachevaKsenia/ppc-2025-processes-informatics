#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace spichek_d_jacobi {

// A * x = b
// A — матрица, b — вектор, eps — точность, max_iter — максимум итераций
using InType = std::tuple<std::vector<std::vector<double>>, std::vector<double>, double, int>;

using OutType = std::vector<double>;

// (input, description)
using TestType = std::tuple<InType, std::string>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace spichek_d_jacobi
