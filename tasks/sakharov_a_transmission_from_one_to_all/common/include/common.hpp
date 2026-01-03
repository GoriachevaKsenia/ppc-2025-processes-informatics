#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace sakharov_a_transmission_from_one_to_all {

using InType = std::tuple<int, std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sakharov_a_transmission_from_one_to_all
