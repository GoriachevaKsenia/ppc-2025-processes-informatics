#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kutuzov_i_torus_grid {

using InType = std::tuple<int, int, std::string>;
using OutType = std::tuple<std::vector<int>, std::string>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kutuzov_i_torus_grid
