#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace leonova_a_radix_merge_sort {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace leonova_a_radix_merge_sort
