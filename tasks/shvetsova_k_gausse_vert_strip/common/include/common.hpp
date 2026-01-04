#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace shvetsova_k_gausse_vert_strip {

using InType = std::pair<std::vector<std::vector<double>>, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shvetsova_k_gausse_vert_strip
