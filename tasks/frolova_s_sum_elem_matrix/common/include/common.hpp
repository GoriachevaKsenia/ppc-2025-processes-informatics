#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace frolova_s_sum_elem_matrix {

using InType = std::tuple<std::vector<double>, int, int>;
using OutType = double;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace frolova_s_sum_elem_matrix
