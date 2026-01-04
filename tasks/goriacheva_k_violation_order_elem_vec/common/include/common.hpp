#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace goriacheva_k_violation_order_elem_vec {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace goriacheva_k_violation_order_elem_vec
