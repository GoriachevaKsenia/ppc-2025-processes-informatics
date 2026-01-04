#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace scalar_product {

using InType = std::pair<std::vector<int>, std::vector<int>>;
using OutType = int;
using TestType = std::tuple<std::pair<std::vector<int>, std::vector<int>>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace scalar_product
