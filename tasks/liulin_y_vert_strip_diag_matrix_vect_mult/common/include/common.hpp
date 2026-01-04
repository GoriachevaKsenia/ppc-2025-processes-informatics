#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace liulin_y_vert_strip_diag_matrix_vect_mult {

using InType = std::tuple<std::vector<std::vector<int>>, std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace liulin_y_vert_strip_diag_matrix_vect_mult
