#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace gusev_d_star {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gusev_d_star
