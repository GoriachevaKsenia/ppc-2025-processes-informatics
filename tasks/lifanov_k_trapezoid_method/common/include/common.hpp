#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace lifanov_k_trapezoid_method {

// Формат входных данных:
// InType = {
//   ax,   // левая граница по X
//   bx,   // правая граница по X
//   ay,   // левая граница по Y
//   by,   // правая граница по Y
//   nx,   // число разбиений по X
//   ny    // число разбиений по Y
// }
using InType = std::vector<double>;

using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace lifanov_k_trapezoid_method
