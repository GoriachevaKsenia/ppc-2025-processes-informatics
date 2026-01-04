#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace leonova_a_star {

using InType = std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>;
using OutType = std::vector<std::vector<int>>;
using TestType =
    std::tuple<std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>, std::vector<std::vector<int>>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace leonova_a_star
