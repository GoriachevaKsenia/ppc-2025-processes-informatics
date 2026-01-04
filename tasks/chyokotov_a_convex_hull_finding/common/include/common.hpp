#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace chyokotov_a_convex_hull_finding {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<std::vector<std::pair<int, int>>>;
using TestType = std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<std::pair<int, int>>>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chyokotov_a_convex_hull_finding
