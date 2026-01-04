#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace chyokotov_a_seidel_method {

using InType = std::pair<std::vector<std::vector<double>>, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<InType, std::vector<double>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chyokotov_a_seidel_method
