#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

using InType = std::vector<int>;
using OutType = std::pair<int, int>;
using TestType = std::tuple<std::pair<int, int>, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nikolaev_d_most_dif_vec_neighbors
