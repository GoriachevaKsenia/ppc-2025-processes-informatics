#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace romanov_a_scatter {

// sendbuf, sendcount, root
using InType = std::tuple<std::vector<int>, int, int>;
// scattered data
using OutType = std::vector<int>;
// sendbuf, sendcount, root
using TestType = std::tuple<std::vector<int>, int, int>;
;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace romanov_a_scatter
