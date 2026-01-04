#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kruglova_a_vertical_ribbon_matvec {

using InType = std::tuple<int, int, std::vector<double>, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kruglova_a_vertical_ribbon_matvec
