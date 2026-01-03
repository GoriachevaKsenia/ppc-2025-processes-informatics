#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace goriacheva_k_reduce {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace goriacheva_k_reduce
